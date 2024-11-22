import os
import time
import random
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.cuda import amp
from spikingjelly import visualizing
import matplotlib.pyplot as plt
import wandb


class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 输出尺寸 14x14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 输出尺寸 7x7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        return x_seq.mean(0)

    def spiking_encoder(self):
        return self.conv_fc[:3]  # 提取前几层用于可视化


def initialize_weights(model, init_method, surrogate_fn):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'lecun':
                fan_in = module.in_features if isinstance(module, nn.Linear) else module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                std = torch.sqrt(torch.tensor(1.0 / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std.item())
            elif init_method.startswith('ikun'):
                init_custom_weights(module, init_method, surrogate_fn)
            else:
                raise ValueError(f"Unsupported initialization method: {init_method}")
    print(f"Initialized model weights using {init_method} method.")


def init_custom_weights(module, init_method, surrogate_fn):
    fan_in = module.in_features if isinstance(module, nn.Linear) else module.in_channels * module.kernel_size[0] * module.kernel_size[1]
    input_values = torch.linspace(-1, 1, steps=1000)
    surrogate_gradient_values = (1 / (1 + input_values ** 2)).abs() if isinstance(surrogate_fn, surrogate.ATan) else None
    avg_gradient = surrogate_gradient_values.mean().item()
    beta = 1.0 / avg_gradient
    std = torch.sqrt(torch.tensor(beta / (fan_in if init_method == 'ikun_v1' else fan_in + module.out_features)))
    nn.init.normal_(module.weight, mean=0.0, std=std.item())
    print(f"Custom initialization ({init_method}): Avg Grad={avg_gradient:.4f}, Std={std:.4f}")


def prepare_data_loaders(args):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root=args.data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False, transform=transform, download=True)

    if args.subset_size:
        all_indices = list(range(len(train_set)))
        random.seed(args.seed)
        fixed_indices = random.sample(all_indices, args.subset_size)
        train_set = torch.utils.data.Subset(train_set, fixed_indices)
        print(f"Using a subset of size {args.subset_size} from training set.")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)

    return train_loader, test_loader


def log_results(epoch, train_loss, train_acc, test_loss, test_acc, optimizer, grad_norm):
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'grad_norm': grad_norm
    })


def main(args_list):
    parser = argparse.ArgumentParser(description='Spiking Neural Network Training with Fashion-MNIST')
    parser.add_argument('-T', type=int, default=4, help='Simulation time-steps')
    parser.add_argument('-device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('-b', type=int, default=128, help='Batch size')
    parser.add_argument('-epochs', type=int, default=64, help='Number of training epochs')
    parser.add_argument('-j', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('-data-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('-out-dir', type=str, default='./logs', help='Output directory')
    parser.add_argument('-opt', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-channels', type=int, default=128, help='Number of channels')
    parser.add_argument('-init', type=str, default='xavier', help='Weight initialization method')
    parser.add_argument('-subset-size', type=int, default=None, help='Subset size for training data')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    args = parser.parse_args(args_list)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_loader, test_loader = prepare_data_loaders(args)
    net = CSNN(T=args.T, channels=args.channels).to(args.device)
    initialize_weights(net, args.init, surrogate.ATan())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) if args.opt == 'adam' else torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    wandb.init(project="spikingjelly_fashion_mnist", config=vars(args), name=f"{args.init}_{args.opt}_{args.lr}")
    wandb.watch(net, log="all")

    for epoch in range(args.epochs):
        net.train()
        train_loss, train_acc, total_samples = 0, 0, 0
        grad_norms = []

        for img, label in train_loader:
            optimizer.zero_grad()
            img, label = img.to(args.device), label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            out_fr = net(img)
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()

            grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in net.parameters() if p.grad is not None]))
            grad_norms.append(grad_norm.item())

            optimizer.step()
            functional.reset_net(net)

            total_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

        train_loss /= total_samples
        train_acc /= total_samples
        avg_grad_norm = sum(grad_norms) / len(grad_norms)

        net.eval()
        test_loss, test_acc, test_samples = 0, 0, 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(args.device), label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                test_loss += F.mse_loss(out_fr, label_onehot).item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                test_samples += label.numel()

        test_loss /= test_samples
        test_acc /= test_samples
        log_results(epoch, train_loss, train_acc, test_loss, test_acc, optimizer, avg_grad_norm)

        lr_scheduler.step()
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    wandb.finish()