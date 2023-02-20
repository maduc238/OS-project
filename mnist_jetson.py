import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision

BATCH_SIZE = 60
EPOCHS = 5
LEARNING_RATE = 0.1
MOMENTUM = 0.5
LOG_INTERVAL = 10

"""Load data"""

from torch.utils.data import Subset

train_data = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data',train=False,download=True,transform=transforms.ToTensor())

train_data = Subset(train_data, indices=range(len(train_data) // 10))
test_data = Subset(test_data, indices=range(len(test_data) // 10))

# change download=True

print('Size of train data:',len(train_data))
print('Size of test data:',len(test_data))

print('The frtst element shape:',train_data[0][0].shape)
print('Label:',train_data[0][1])

"""Create data loader"""

from torch.utils.data import DataLoader

train_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE)

"""from torch.utils.data import DataLoader

train_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)
"""

class SimpleNetwork(nn.Module):
    def __init__(self):  # cấu trúc mạng
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):   # dataflow
        z1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        z2 = self.conv2(z1)
        z3 = F.relu(F.max_pool2d(self.conv2_drop(z2), 2))
        z3 = z3.view(-1, 320)
        z4 = F.dropout(F.relu(self.fc1(z3)))
        z5 = self.fc2(z4)
        return F.log_softmax(z5, dim=1)

"""Train and test model"""

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))
def test():
    # evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))

model = SimpleNetwork()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test()