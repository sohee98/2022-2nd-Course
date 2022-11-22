'''
    2022.11.18. EE7107 @ KENTECH
    Image classification with AlexNet

    Requirements:
    Please install PyTorch in your local server.
    https://pytorch.org/
    Stable (1.13.0) >> Linux >> Conda >> Python >> CUDA xx.x
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torchvision import transforms
from matplotlib import pyplot as plt
import pdb
import torchvision.datasets as datasets
import torch.optim as optim


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # [Q] Please fill in the input parameters below:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)     # input 1 channel image (MNIST_Fashion)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)   # 256*6*6=9216
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
    
    def forward(self, x):
        # [Q] (Use pdb) Please answer the shape of following tensors:
        # pdb.set_trace()
        # x.shape: torch.Size([1, 1, 227, 227])
        x = F.relu(self.conv1(x))
        # x.shape: torch.Size([1, 96, 55, 55])
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x.shape: torch.Size([1, 96, 27, 27])
        x = F.relu(self.conv2(x))
        # x.shape: torch.Size([1, 256, 27, 27])
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x.shape: torch.Size([1, 256, 13, 13])
        x = F.relu(self.conv3(x))
        # x.shape: torch.Size([1, 384, 13, 13])
        x = F.relu(self.conv4(x))
        # x.shape: torch.Size([1, 384, 13, 13])
        x = F.relu(self.conv5(x))
        # x.shape: torch.Size([1, 256, 13, 13])
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x.shape: torch.Size([1, 256, 6, 6])
        x = x.view(x.size(0), -1) # flatten
        # x.shape: torch.Size([1, 9216])
        
        x = F.relu(self.fc1(x))
        # x.shape: torch.Size([1, 4096])
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        # x.shape: torch.Size([1, 4096])
        x = F.dropout(x, p=0.5)
        x = F.log_softmax(self.fc3(x), dim=1)
        # x.shape: torch.Size([1, 1000])
        
        return x 
        
    @staticmethod
    def transform():
        return transforms.Compose([transforms.Resize((227, 227)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        '''
            [Q] What does "Normalize()" mean?
                to make values' range to 0 ~ 1
                the values become small
        '''


if __name__=="__main__":
    # if gpu is to be used
    use_cuda = torch.cuda.is_available()
    print("use_cuda : ", use_cuda)
    
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device= torch.device("cuda:0" if use_cuda else "cpu")
    
    net = AlexNet().to(device)
    
    # forward-pass random variable
    X = torch.randn(size=(1, 1, 227, 227)).type(FloatTensor)
    print(net(X))
    # print(summary(net, (1, 227, 227))) 


    # hyper parameter
    batch_size = 512
    num_epochs = 20
    # num_epochs = 10
    learning_rate = 0.0001
    '''
        [Q] What is the batch size?
            the number of training examples present in a single batch 

        [Q] What is the epoch?
            one epoch look at all dataset one time. it is number of times how much to train all dataset.

        [Q] Please discuss the relationship between the number of iterations (for an epoch) and the batch size.
            iterations(반복횟수) are number of training times. iterations*batch size is the number of data size.


    '''
    
    # data load
    root = './MNIST_Fashion'
    transform = AlexNet.transform()
    train_set = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True) 

    
    # if gpu is to be used 
    use_cuda = torch.cuda.is_available() 
    print("use_cuda : ", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = AlexNet().to(device)
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 30 == 0:
                print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        return loss.item()
                
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target, reduction='sum').item() 
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            print('='*50)    
        return test_loss

    train_loss = []
    test_loss = []

    for epoch in range(1, num_epochs + 1):
        train_l = train(model, device, train_loader, optimizer, epoch)
        test_l = test(model, device, test_loader)
        train_loss.append(train_l)
        test_loss.append(test_l)

    pdb.set_trace() 

    '''
        [Q] Please plot the loss & accuracy graph for both training and test phase.
        (Pdb) train_loss
        [0.5456092953681946, 0.5422663688659668, 0.2426520586013794, 0.2701944410800934, 0.47821056842803955, 0.16200780868530273, 0.174241840839386, 0.18932662904262543, 0.18114042282104492, 0.22142712771892548]
        (Pdb) test_loss
        [0.5964089431762696, 0.43458033752441405, 0.3583654022216797, 0.33160087890625, 0.29789561157226563, 0.28953544540405274, 0.27427194290161133, 0.2685143222808838, 0.26199447555541994, 0.24841377487182617]

        
    '''

    plt.figure(figsize=(10,10))
    plt.plot(train_loss,'b',label='train')
    plt.plot(test_loss,'r',label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss / Test loss')
    plt.legend(loc='upper right')
    plt.show()
