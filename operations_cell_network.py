# operations, cell and network are based on the previous work: https://github.com/chenxi116/PNASNet.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)
        )
        
    def forward(self, x):
        return self.op(x)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        
    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

class SepConv(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.operation = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=channels, bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                      stride=1, padding=padding, groups=channels, bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=channels),
        )
        
    def forward(self, x):
        return self.operation(x)
    
class DilatedSepConv(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation):
        super(DilatedSepConv, self).__init__()
        self.operation = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=channels, dilation=dilation, bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=channels),
        )
        
    def forward(self, x):
        return self.operation(x)
    
class MaxPool(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(MaxPool, self).__init__()
        self.operation = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        )
        
    def forward(self, x):
        return self.operation(x)
            
class AvgPool(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(AvgPool, self).__init__()
        self.operation = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        )
        
    def forward(self, x):
        return self.operation(x)
    
class Identity(nn.Module):
    def __init__(self, channels, stride):
        super(Identity, self).__init__()
        layers = [nn.Identity()]
        if stride != 1:
            layers = [ReLUConvBN(C_in=channels, C_out=channels, kernel_size=1, stride=stride, padding=0)]
        self.operation = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.operation(x)
    
class Zero(nn.Module):
    def __init__(self, channels, stride):
        super(Zero, self).__init__()
        self.stride = stride
        
    def forward(self, x):
        return torch.mul(x[:, :, ::self.stride, ::self.stride], 0)
    
operations = {'sep_conv_3x3': lambda channels, stride: SepConv(channels, 3, stride, 1),
              'sep_conv_5x5': lambda channels, stride: SepConv(channels, 5, stride, 2),
              'dil_sep_conv_3x3': lambda channels, stride: DilatedSepConv(channels, 3, stride, 2, 2),
              'dil_sep_conv_5x5': lambda channels, stride: DilatedSepConv(channels, 5, stride, 4, 2),
              'max_pool_3x3': lambda channels, stride: MaxPool(channels, 3, stride, 1),
              'avg_pool_3x3': lambda channels, stride: AvgPool(channels, 3, stride, 1),
              'identity': lambda channels, stride: Identity(channels, stride),
              'zero': lambda channels, stride: Zero(channels, stride),
             }

def mixed_operation(x, op_dict, alpha):
    prob = F.softmax(alpha, dim=0)
    out = sum([prob[i] * op_dict[op_name](x) for i, op_name in enumerate(op_dict)])
    return out

class Cell(nn.Module):
    def __init__(self, channels, stride):
        super(Cell, self).__init__()
        self.n_nodes = 6
        self.n_groups = self.n_nodes - 3
        self.graph_op_dict = nn.ModuleDict()
        for j in range(2, self.n_nodes - 1):
            for i in range(j):
                cur_stride = 1
                if i in [0, 1]:
                    cur_stride = stride
                op_dict = nn.ModuleDict()
                for op_name in operations:
                    op_dict.add_module(op_name, operations[op_name](channels, cur_stride))
                self.graph_op_dict.add_module('{}_{}'.format(i, j), op_dict)
                
    def forward(self, x0, x1, alpha_dict):
        inputs = [x0, x1]
        for j in range(2, self.n_nodes - 1):
            mixed_op_outs = []
            for i in range(j):
                mixed_op_outs.append(mixed_operation(inputs[i],
                                                     self.graph_op_dict['{}_{}'.format(i, j)],
                                                     alpha_dict['{}_{}'.format(i, j)]))
            inputs.append(sum(mixed_op_outs))
        out = torch.concat(inputs[2:], dim=1)
        return out
    
class Net(nn.Module):
    def __init__(self, n_layers, channels, hidden_sizes, n_classes=10):
        super(Net, self).__init__()
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        self.adjust_input_0 = nn.ModuleList()
        self.adjust_input_1 = nn.ModuleList()
        channels_input_0 = 3
        channels_input_1 = 3
        reduction_cur = False
        reduction_prev = False
        
        for i in range(n_layers):
            if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
                reduction_cur = True
                stride = 2
            else:
                reduction_cur = False
                stride = 1
            
            if reduction_prev:
                channels *= 2
                self.adjust_input_0.add_module(str(i), FactorizedReduce(C_in=channels_input_0, C_out=channels))
            else:
                self.adjust_input_0.add_module(str(i), ReLUConvBN(C_in=channels_input_0, C_out=channels, kernel_size=1, stride=1, padding=0))
            self.adjust_input_1.add_module(str(i), ReLUConvBN(C_in=channels_input_1, C_out=channels,
                                             kernel_size=1, stride=1, padding=0))
        
            cur_cell = Cell(channels=channels, stride=stride)
            self.cells.add_module(str(i), cur_cell)

            reduction_prev = reduction_cur

            channels_input_0 = channels_input_1
            channels_input_1 = channels * cur_cell.n_groups
            
        in_features = ((32 // (2 ** 2)) ** 2) * channels * cur_cell.n_groups
        clf_layers = [nn.Flatten()]
        for i in range(len(hidden_sizes)):
            out_features = hidden_sizes[i]
            clf_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            clf_layers.append(nn.ReLU())
            in_features = out_features
        clf_layers.append(nn.Linear(in_features=in_features, out_features=n_classes))
        self.classifier = nn.Sequential(*clf_layers)
        
    def forward(self, x, alpha_normal_dict, alpha_reduce_dict):
        input_0 = x
        input_1 = x
        for i in range(self.n_layers):
            input_0_adjusted = self.adjust_input_0[i](input_0)
            input_1_adjusted = self.adjust_input_1[i](input_1)
            if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
                cell_out = self.cells[i](input_0_adjusted, input_1_adjusted, alpha_reduce_dict)
            else:
                cell_out = self.cells[i](input_0_adjusted, input_1_adjusted, alpha_normal_dict)
            input_0 = input_1
            input_1 = cell_out
        out = self.classifier(cell_out)
        return out

class LearnedCell(nn.Module):
    def __init__(self, channels, stride, edge2oper_dict):
        super(LearnedCell, self).__init__()
        self.n_nodes = 6
        self.n_groups = self.n_nodes - 3
        self.graph_op_dict = nn.ModuleDict()
        for edge_name in edge2oper_dict:
            i, j = map(int, edge_name.split('_'))
            cur_stride = 1
            if i in [0, 1]:
                cur_stride = stride
            for k, op_name in enumerate(operations):
                if k == edge2oper_dict[edge_name]:
                    self.graph_op_dict.add_module(edge_name, operations[op_name](channels, cur_stride))
        
    def forward(self, x0, x1):
        inputs = [x0, x1]
        for j in range(2, self.n_nodes - 1):
            node_outs = []
            for i in range(j):
                edge_name = '{}_{}'.format(i, j)
                if edge_name in self.graph_op_dict:
                    node_outs.append(self.graph_op_dict[edge_name](inputs[i]))
            inputs.append(sum(node_outs))
        out = torch.concat(inputs[2:], dim=1)
        return out
    
class LearnedNet(nn.Module):
    def __init__(self, n_layers, channels, hidden_sizes, edge2oper_normal, edge2oper_reduce, n_classes=10):
        super(LearnedNet, self).__init__()
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        self.adjust_input_0 = nn.ModuleList()
        self.adjust_input_1 = nn.ModuleList()
        channels_input_0 = 3
        channels_input_1 = 3
        reduction_cur = False
        reduction_prev = False
        
        for i in range(n_layers):
            if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
                reduction_cur = True
                stride = 2
            else:
                reduction_cur = False
                stride = 1
            
            if reduction_prev:
                channels *= 2
                self.adjust_input_0.add_module(str(i), FactorizedReduce(C_in=channels_input_0, C_out=channels))
            else:
                self.adjust_input_0.add_module(str(i), ReLUConvBN(C_in=channels_input_0, C_out=channels, kernel_size=1, stride=1, padding=0))
            self.adjust_input_1.add_module(str(i), ReLUConvBN(C_in=channels_input_1, C_out=channels,
                                             kernel_size=1, stride=1, padding=0))
        
            cur_cell = LearnedCell(channels=channels, stride=stride,
                                   edge2oper_dict=[edge2oper_normal, edge2oper_reduce][reduction_cur])
            self.cells.add_module(str(i), cur_cell)

            reduction_prev = reduction_cur

            channels_input_0 = channels_input_1
            channels_input_1 = channels * cur_cell.n_groups
            
        in_features = ((32 // (2 ** 2))**2) * channels * cur_cell.n_groups
        clf_layers = [nn.Flatten()]
        for i in range(len(hidden_sizes)):
            out_features = hidden_sizes[i]
            clf_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            clf_layers.append(nn.ReLU())
            in_features = out_features
        clf_layers.append(nn.Linear(in_features=in_features, out_features=n_classes))
        self.classifier = nn.Sequential(*clf_layers)
        
    def forward(self, x):
        input_0 = x
        input_1 = x
        for i in range(self.n_layers):
            input_0_adjusted = self.adjust_input_0[i](input_0)
            input_1_adjusted = self.adjust_input_1[i](input_1)
            if i in [self.n_layers // 3, 2 * self.n_layers // 3]:
                cell_out = self.cells[i](input_0_adjusted, input_1_adjusted)
            else:
                cell_out = self.cells[i](input_0_adjusted, input_1_adjusted)
            input_0 = input_1
            input_1 = cell_out
        out = self.classifier(cell_out)
        return out