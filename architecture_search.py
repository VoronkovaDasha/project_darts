import numpy as np
import torchvision
from tqdm import tqdm
import argparse
import torch
from operations_cell_network import *
import os
import csv

def update_architecture(model, optim_model, model_helper, optim_model_helper,
                        alpha_normal, optim_alpha_normal,
                        alpha_reduce, optim_alpha_reduce,
                        train_inputs, train_labels,
                        val_inputs, val_labels,
                        criterion, use_xi):

    model_helper.load_state_dict(model.state_dict())
    model_helper.train()
    optim_model_helper.load_state_dict(optim_model.state_dict())
    if use_xi:
        xi = optim_model_helper.param_groups[-1]['lr']

    # compute w_prime
    if use_xi:
        train_out = model_helper(train_inputs, alpha_normal, alpha_reduce)
        L_train_w_alpha = criterion(train_out, train_labels)
        optim_model_helper.zero_grad()
        L_train_w_alpha.backward()
        optim_model_helper.step()
    
    # grad of L_val_w_prime_alpha w.r.t alpha
    val_out = model_helper(val_inputs, alpha_normal, alpha_reduce)
    L_val_w_prime_alpha = criterion(val_out, val_labels)
    optim_model_helper.zero_grad()
    optim_alpha_normal.zero_grad()
    optim_alpha_reduce.zero_grad()
    L_val_w_prime_alpha.backward()
    
    if use_xi:
        # compute eps
        grads = [p.grad.detach().flatten() for p in model_helper.parameters() if p.grad is not None]
        grad_norm = torch.cat(grads).norm()
        eps = 0.01 / grad_norm
        
        # w_plus and loss backward
        for p1, p2 in zip(model.parameters(), model_helper.parameters()):
            p1.data += eps * p2.grad.data
        train_out_plus = model(train_inputs, alpha_normal, alpha_reduce)
        L_train_w_plus_alpha = -xi * criterion(train_out_plus, train_labels) / 2 / eps
        L_train_w_plus_alpha.backward()
        
        # w_minus and loss backward
        for p1, p2 in zip(model.parameters(), model_helper.parameters()):
            p1.data -= 2 * eps * p2.grad.data    # back to original weights and then to w_minus
        train_out_minus = model(train_inputs, alpha_normal, alpha_reduce)
        L_train_w_minus_alpha = xi * criterion(train_out_minus, train_labels) / 2 / eps
        L_train_w_minus_alpha.backward()

        # restore weights
        for p1, p2 in zip(model.parameters(), model_helper.parameters()):
            p1.data += eps * p2.grad.data
            
    # update alpha
    optim_alpha_normal.step()
    optim_alpha_reduce.step()

parser = argparse.ArgumentParser(description='DARTS project')
parser.add_argument('--n_channels', type=int, default=16,
                    help='initial number of channels in the cell')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--use_xi', action='store_true',
                    help='whether to use first order approximation')

args = parser.parse_args()

# Prepare datasets
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

indices = np.arange(len(dataset))
n_train = len(indices) // 2
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

# Prepare alpha and networks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ('device {}'.format(device))

criterion = nn.CrossEntropyLoss()

# use 6 nodes instead of 7
n_nodes = 6
alpha_normal_dict = nn.ParameterDict({'{}_{}'.format(i, j): nn.Parameter(data=torch.zeros(len(operations)), requires_grad=True) 
                                      for j in range(2, n_nodes - 1) for i in range(j)})
alpha_reduce_dict = nn.ParameterDict({'{}_{}'.format(i, j): nn.Parameter(data=torch.zeros(len(operations)), requires_grad=True) 
                                      for j in range(2, n_nodes - 1) for i in range(j)})
alpha_normal_dict.to(device)
alpha_reduce_dict.to(device)

# use values reported in the paper
net = Net(8, args.n_channels, [2048, 1024, 256])
print ('Net used:')
print (net)
net.to(device)
net_helper = Net(8, args.n_channels, [2048, 1024, 256])
net_helper.to(device)
net_helper.load_state_dict(net.state_dict())
net_helper.train()

# use values reported in the paper
optim_net = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
optim_net_helper = torch.optim.SGD(net_helper.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
optim_alpha_normal = torch.optim.Adam(alpha_normal_dict.parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
optim_alpha_reduce = torch.optim.Adam(alpha_reduce_dict.parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_net, T_max=args.n_epochs)

# save training progress
columns = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
stats_file_path = 'train_stats.csv'
if not os.path.exists(stats_file_path):
    with open(stats_file_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

# save alpha history
alpha_normal_history = np.zeros((args.n_epochs, len(alpha_normal_dict), alpha_normal_dict['0_2'].shape[0]))
alpha_reduce_history = np.zeros((args.n_epochs, len(alpha_reduce_dict), alpha_reduce_dict['0_2'].shape[0]))

for epoch in range(args.n_epochs):
    train_loss = 0.
    train_acc = 0.
    total = 0
    for train_inputs, train_labels in tqdm(train_loader):
        net.train()
        
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        
        val_inputs, val_labels = next(iter(val_loader))
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        
        # alpha optimization first
        update_architecture(net, optim_net, net_helper, optim_net_helper,
                            alpha_normal_dict, optim_alpha_normal,
                            alpha_reduce_dict, optim_alpha_reduce,
                            train_inputs, train_labels,
                            val_inputs, val_labels,
                            criterion, args.use_xi)
        
        # parameters optimization second
        out = net(train_inputs, alpha_normal_dict, alpha_reduce_dict)
        loss = criterion(out, train_labels)
        optim_net.zero_grad()
        loss.backward()
        optim_net.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += train_labels.shape[0]
        train_acc += (predicted == train_labels).sum().item()
    train_loss /= total
    train_acc /= total
    print ('Epoch {}: Train Loss {}, Train Accuracy {}'.format(epoch, train_loss, train_acc))
    
    # validation
    val_loss = 0.
    val_acc = 0.
    total = 0.
    net.eval()
    for val_inputs, val_labels in val_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_out = net(val_inputs, alpha_normal_dict, alpha_reduce_dict)
        val_loss += criterion(val_out, val_labels).item()
        _, predicted = torch.max(val_out.data, 1)
        total += val_labels.shape[0]
        val_acc += (predicted == val_labels).sum().item()
    val_loss /= total
    val_acc /= total
    print ('Epoch {}: Val Loss {}, Val Accuracy {}'.format(epoch, val_loss, val_acc))
    print ('Alpha normal:')
    for p in alpha_normal.parameters():
        print (F.softmax(p))
    print ('Alpha reduce:')
    for p in alpha_reduce.parameters():
        print (F.softmax(p))
    
    print ()
        
    # scheduler
    scheduler.step()
    
    with open(stats_file_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        row_dict = {'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc}
        writer.writerow(row_dict)
        
    for i, edge in enumerate(alpha_normal_dict):
        alpha_normal_history[epoch, i] = alpha_normal_dict[edge].detach().cpu().data.numpy()
    np.save('alpha_normal_history.npy', alpha_normal_history)
    
    for i, edge in enumerate(alpha_reduce_dict):
        alpha_reduce_history[epoch, i] = alpha_reduce_dict[edge].detach().cpu().data.numpy()
    np.save('alpha_reduce_history.npy', alpha_reduce_history)
