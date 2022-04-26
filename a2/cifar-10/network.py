'''
JOB LOG:
采用和 handout 一样的层级架构但参数不同。
一开始参数少，准确率低。尝试多加一层 conv + relu，并没有显著的效果；
又修改为增加每层的参数，特别是增加通道数和增大 kernel size，效果显著。
'''
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 16*16
            nn.Conv2d(64, 64, 5, padding=1, stride=1),  # 14*14
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=1, stride=1),  # 12*12
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),  # 4*4
            nn.Flatten(),  # 4*4*64
            nn.Linear(4 * 4 * 64, 512),  # 512
            nn.ReLU(),
            nn.Linear(512, 10)  # 10
        )

    def forward(self, x):
        # x: B*H*W*3
        logits = self.seq(x)  # B*N
        return logits


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from dataset import CIFAR10

    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break
