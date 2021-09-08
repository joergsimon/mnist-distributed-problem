import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

import sys
sys.path.append("../")
print(torch.__version__)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Declare transform to convert raw data to tensor
transforms = transforms.Compose([
                                transforms.ToTensor()
])

# Loading Data and splitting it into train and validation data
train = datasets.MNIST('', train = True, transform = transforms, download = True)
train, valid = random_split(train,[50000,10000])

# Create Dataloader of the above tensor with batch size = 32
trainloader = DataLoader(train, batch_size=32, pin_memory=True, num_workers=0)
validloader = DataLoader(valid, batch_size=32, pin_memory=True, num_workers=0)

# Building Our Mode
class Network(nn.Module):
    # Declaring the Architecture
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    # Forward Pass
    def forward(self, x):
        x = x.view(x.shape[0],-1) # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Network()
if torch.cuda.is_available():
    model = model.cuda()

# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.div_(batch_size))
        return res

# Training with Validation


def main():
    epochs = 5
    for e in range(epochs):

        min_valid_loss = np.inf
        train_losses = []
        valid_losses = []
        valid_accuracies = []

        model.train()
        for data, labels in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data, labels in validloader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                pred = model(data)
                loss = criterion(pred, labels)
                valid_losses.append(loss.item())

                valid_accuracies.append(accuracy(pred, labels)[0].item())

        train_loss = np.mean(np.array(train_losses))
        valid_loss = np.mean(np.array(valid_losses))
        valid_accuracy = np.mean(np.array(valid_accuracies))

        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')

    print(f'final valid accuracy: {valid_accuracy}')


if __name__ == '__main__':
    main()