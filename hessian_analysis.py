import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pyhessian import hessian
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import matplotlib.pyplot as plt
import shutil
import csv

# 配置全局绘图字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# 定义CSNN模型
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 输出尺寸: 14x14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 输出尺寸: 7x7

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
        # 扩展时间维度并重复T次
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        return x_seq.mean(0)  # 返回发放率

    def spiking_encoder(self):
        return self.conv_fc[0:3]  # 返回编码部分


# 处理模型权重并计算Hessian特征值
def process_model_weights(weight_path, model_name, output_dir, device='cuda'):
    T = 4
    channels = 128
    use_cupy = False
    batch_size = 1
    data_dir = './data/'

    # 加载模型
    model = CSNN(T=T, channels=channels, use_cupy=use_cupy).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    # 加载测试集
    test_set = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, transform=transforms.ToTensor(), download=True
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # 获取一批测试数据
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    targets_onehot = F.one_hot(targets, 10).float()

    # Hessian计算
    criterion = F.mse_loss
    hessian_comp = hessian(model, criterion, data=(inputs, targets_onehot), cuda=(device == 'cuda'))

    # 获取前50个Hessian特征值
    top_k = 50
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_k)
    print(f"模型 {model_name} 的前{top_k}个Hessian特征值已计算。")

    # 计算Hessian的迹
    trace = hessian_comp.trace()
    print(f"模型 {model_name} 的Hessian迹为: {np.mean(trace):.4f}")

    return top_eigenvalues


# 主函数
def main(log_dir='log', output_dir='output', device='cuda'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eigenvalues_dict = {}

    # 遍历log目录下的模型权重
    for model_name in os.listdir(log_dir):
        model_dir = os.path.join(log_dir, model_name)
        if os.path.isdir(model_dir):
            weight_path = os.path.join(model_dir, 'checkpoint_max.pth')
            if os.path.exists(weight_path):
                print(f"处理模型 '{model_name}' 的权重文件: {weight_path}")
                try:
                    eigenvalues = process_model_weights(weight_path, model_name, output_dir, device)
                    eigenvalues_dict[model_name] = eigenvalues
                except Exception as e:
                    print(f"处理模型 '{model_name}' 时发生错误: {e}")
            else:
                print(f"未找到模型 '{model_name}' 的权重文件。")

    # 保存特征值数据到CSV
    csv_path = os.path.join(output_dir, "hessian_eigenvalues.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ["特征值排序"] + list(eigenvalues_dict.keys())
        writer.writerow(header)

        max_length = max(len(eigenvalues) for eigenvalues in eigenvalues_dict.values())
        for i in range(max_length):
            row = [i + 1]
            for model_name in eigenvalues_dict.keys():
                row.append(eigenvalues_dict[model_name][i] if i < len(eigenvalues_dict[model_name]) else "")
            writer.writerow(row)

    print(f"特征值数据已保存到 {csv_path}")

    # 绘制前50个Hessian特征值对比图
    plt.figure(figsize=(12, 8))
    for model_name, eigenvalues in eigenvalues_dict.items():
        x = np.arange(1, min(50, len(eigenvalues)) + 1)
        plt.plot(x, eigenvalues[:50], label=model_name)
    plt.xlabel('特征值排序')
    plt.ylabel('特征值大小')
    plt.title('各模型的前50个Hessian特征值对比')
    plt.legend(fontsize='small')
    plt.grid(True)

    comparison_plot_path = os.path.join(output_dir, "hessian_eigenvalues_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"特征值对比图已保存到 {comparison_plot_path}")


if __name__ == '__main__':
    main()
