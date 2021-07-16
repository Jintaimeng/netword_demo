import torch
#net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)
    def __format__(self, x):
        out = self.CNN(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
