import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(16 * 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 16 * 1)
        x = self.fc(x)
        return x

class SimplifiedVGG16(nn.Module):
    ''''
    VGG model 16 会使得输入数据的长度在经过一系列卷积和池化操作后变得过小，使得池化操作无法正常进行。
    '''
    def __init__(self):
        super(SimplifiedVGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)
        )
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128)
        x = self.classifier(x)
        return x
    
class IrisDataset(Dataset):
    def __init__(self, type='train'):
        self.type = type
        
        if type == 'train':
            data = np.genfromtxt('iris_training.csv', delimiter=',', skip_header=1, dtype=np.float32)

            train_x = torch.tensor(data[:, :-1])
            train_y = torch.tensor(data[:, -1], dtype=torch.int64)
            self.X = train_x.view(-1, 1, 4)
            self.y = train_y
        elif type == 'test':
            data = np.genfromtxt('iris_test.csv', delimiter=',', skip_header=1, dtype=np.float32)

            test_x = torch.tensor(data[:, :-1])
            test_y = torch.tensor(data[:, -1], dtype=torch.int64)
            self.X = test_x.view(-1, 1, 4)
            self.y = test_y
        else:
            raise ValueError("Invalid type. Choose 'train' or 'test'.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train(model, train_loader, criterion, optimizer, num_epochs):

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')   

def test(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples =0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print(f'Test Accuracy: {accuracy * 100:.2f}%')

def main():
    train_dataset = IrisDataset(type='train')
    test_dataset = IrisDataset(type='test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN()
    '''
    训练集上的准确率最大为97.50%，测试集上的准确率为96.67%
    '''
    # model = SimplifiedVGG16()
    '''
    训练集上的准确率最大为100.00%，测试集上的准确率为96.67%
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 800

    train(model, train_loader, criterion, optimizer, num_epochs)    
    test(model, test_loader)    

if __name__ == "__main__":
    main()
