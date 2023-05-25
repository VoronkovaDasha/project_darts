import argparse
import csv
import torchvision
import torch
import torch.nn as nn
import numpy as np
from operations_cell_network import *
import os
from tqdm import tqdm
import torch.nn.functional as F

def get_edge2oper(alpha_dict, n_nodes, operations):
    non_zero_index = []
    for i, op_name in enumerate(operations):
        if op_name != 'zero':
            non_zero_index.append(i)
    edge2oper = {}
    top_k = 2
    for j in range(2, n_nodes-1):
        node_incom = {}
        for i in range(j):
            edge_name = '{}_{}'.format(i, j)
            alpha = F.softmax(alpha_dict[edge_name])
            node_incom[i] = (torch.argmax(alpha[non_zero_index]).item(), torch.max(alpha[non_zero_index]).item())
        node_incom = dict(sorted(node_incom.items(), key=lambda x: x[-1][-1], reverse=True))
        for n, i in enumerate(node_incom):
            if n < top_k:
                edge_name = '{}_{}'.format(i, j)
                edge2oper[edge_name] = node_incom[i][0]
    return edge2oper

def load_alpha(alpha_dict, alpha_path, epoch):
    alpha_dict_ckpt = np.load(alpha_path)[epoch-1]
    for i, edge in enumerate(alpha_dict):
        alpha_dict[edge].data = torch.from_numpy(alpha_dict_ckpt[i])
        
def train(model, optimizer, loader, criterion, device):
    train_loss = 0.
    train_acc = 0.
    total = 0
    model.train()
    for (inputs, labels) in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        out = model(inputs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += labels.shape[0]
        train_acc += (predicted == labels).sum().item()
    train_loss /= total
    train_acc /= total
    
    return train_loss, train_acc

def eval(model, loader, criterion, device):
    val_loss = 0.
    val_acc = 0.
    total = 0
    model.eval()
    for (inputs, labels) in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        out = model(inputs)
        loss = criterion(out, labels)
        
        val_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += labels.shape[0]
        val_acc += (predicted == labels).sum().item()
    val_loss /= total
    val_acc /= total
    
    return val_loss, val_acc

parser = argparse.ArgumentParser(description='DARTS project')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset: CIFAR10 or CIFAR100')
parser.add_argument('--n_epochs', type=int, default=600,
                    help='number of epochs')
parser.add_argument('--alpha_normal_path', type=str, required=True,
                    help='path to alpha normal history')
parser.add_argument('--alpha_reduce_path', type=str, required=True,
                    help='path to alpha reduce history')
parser.add_argument('--alpha_epoch', type=int, default=40,
                    help='epoch to load alphas')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size for training')

args = parser.parse_args()

# Prepare datasets
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if args.dataset == 'CIFAR10':
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    n_classes = 10
elif args.dataset == 'CIFAR100':
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    n_classes = 100

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

# Prepare alpha and networks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ('device {}'.format(device))

criterion = nn.CrossEntropyLoss()

n_nodes = 6
alpha_normal_dict = nn.ParameterDict({'{}_{}'.format(i, j): nn.Parameter(data=torch.zeros(len(operations))) 
                                      for j in range(2, n_nodes - 1) for i in range(j)})
alpha_reduce_dict = nn.ParameterDict({'{}_{}'.format(i, j): nn.Parameter(data=torch.zeros(len(operations))) 
                                      for j in range(2, n_nodes - 1) for i in range(j)})
load_alpha(alpha_normal_dict, alpha_path=args.alpha_normal_path, epoch=args.alpha_epoch)
load_alpha(alpha_reduce_dict, alpha_path=args.alpha_reduce_path, epoch=args.alpha_epoch)

edge2oper_normal = get_edge2oper(alpha_normal_dict, n_nodes, operations)
print ('edge to operation normal')
print (edge2oper_normal)
edge2oper_reduce = get_edge2oper(alpha_reduce_dict, n_nodes, operations)
print ('edge to operation reduce')
print (edge2oper_reduce)

net = LearnedNet(14, 16, [4096, 1024, 256], edge2oper_normal, edge2oper_reduce, n_classes)
print ('Net used:')
print (net)
net.to(device)

# use values reported in the paper
optim_net = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_net, T_max=args.n_epochs)

columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
stats_file_path = 'net_train_stats_{}.csv'.format(args.dataset)
if not os.path.exists(stats_file_path):
    with open(stats_file_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        
for epoch in range(args.n_epochs):
    train_loss, train_acc = train(net, optim_net, train_loader, criterion, device)
    print ('Epoch {}: Train Loss {}, Train Accuracy {}'.format(epoch, train_loss, train_acc))
    test_loss, test_acc = eval(net, test_loader, criterion, device)
    print ('Epoch {}: Test Loss {}, Test Accuracy {}'.format(epoch, test_loss, test_acc))
        
    # scheduler
    scheduler.step()
    
    with open(stats_file_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        row_dict = {'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc}
        writer.writerow(row_dict)
        