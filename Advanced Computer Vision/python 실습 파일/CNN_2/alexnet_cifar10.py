'''
    2022.11.29. EE7107 @ KENTECH
    Train/test AlexNet with CIFAR-10 dataset
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pdb


# Visualization module for the feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

transform = transforms.Compose([
    transforms.Resize(256),     # 256의 크기로 이미지를 늘린다 -> Data augmentation
    transforms.CenterCrop(224), # 224만큼 중앙에서부터 자른다 -> Input for AlexNet
    transforms.ToTensor(),      # 데이터셋을 Tensor 정태로 변환(동시에 값이 0 ~ 1로 정규화됨)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])   # normalization
])

batch_size = 500
epoch_size = 5

# Download CIFAR-10 training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 50000(train dataset size) / 500(batch_size) = 100번의 iteration을 수행하면 모든 데이터를 다 볼 수 있음
# shuffle = True: Randomly shuffled dataset, False: original sequence 
# num_workers = multi-thread processes
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

# Download CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# shuffle = False
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Assigned classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# [Q1] Please explain the "weights" parameter. => get weights from Imagenet pretrained model
# model = torchvision.models.alexnet(weights=None)
model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

# Assign the input & output number of channels for each FC layer: 9216 -> 4096 -> 1024 -> 10
model.classifier[4] = nn.Linear(4096,1024)
model.classifier[6] = nn.Linear(1024,10)
                                        
# Layer definition
print(model.eval())
# pdb.set_trace()
'''
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=1024, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=1024, out_features=10, bias=True)
  )
)
'''

# CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# GPU computing
model = model.to(device)

model.features[0].register_forward_hook(get_activation('conv1'))
model.features[3].register_forward_hook(get_activation('conv2'))
model.features[6].register_forward_hook(get_activation('conv3'))
model.features[8].register_forward_hook(get_activation('conv4'))
model.features[10].register_forward_hook(get_activation('conv5'))

# Loss function: Cross entropy loss
criterion = nn.CrossEntropyLoss()

# SGD or Adam optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)



##############################################
############### Training Phase ###############

for epoch in range(epoch_size):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):   # num of iters = data_size / batch_size
        # Decouple the parsed data
        inputs, labels = data

        # For GPU computing
        inputs = inputs.to(device)          # torch.Size([500, 3, 224, 224])
        labels = labels.to(device)
        # pdb.set_trace()
        '''
            [Q2] Visualize inputs using pyplot library. Why do we need re-normalization for the visualization?
                => inputs rgb values are not correct when it is normalized. (they are 0~1)
                   so we need to re-normalize to make values rgb correctly. (0~255)
            
            plt.figure(1); plt.imshow(inputs[0].cpu().permute(1,2,0)); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(2); plt.imshow(inputs[0].cpu().numpy().transpose(1,2,0)); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(3); plt.imshow(inputs[0].cpu().numpy().transpose(1,2,0)*0.225+0.5); plt.colorbar(); plt.ion(); plt.show()

            <Hints>
            >> plt.figure(1); plt.imshow("image array or tensor"); plt.colorbar(); plt.ion(); plt.show()
            >> image tensor for the first batch: inputs[0].cpu().permute(1,2,0)
            >> image array for the first batch: inputs[0].cpu().numpy().transpose(1,2,0)
            >> with re-normalization: inputs[0].cpu().numpy().transpose(1,2,0)*0.225+0.5

        '''

        # Make the gradients zeros
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # pdb.set_trace()

        # if epoch == 0:  pdb.set_trace()
        # if epoch == 0 and i == 0:  
        #     plt.figure(figsize=(10, 10)) 
        #     for k in range(64):
        #         plt.subplot(8,8,k+1); plt.imshow(model.features[0].weight.data.clone()[k].mean(dim=0).cpu(), cmap='bone')
        #     plt.show()
        #     pdb.set_trace()

        # if epoch == 4:  pdb.set_trace()
        if epoch == 4 and i == 99:  
            plt.figure(figsize=(10, 10)) 
            for k in range(64):
                plt.subplot(8,8,k+1); plt.imshow(model.features[0].weight.data.clone()[k].mean(dim=0).cpu(), cmap='bone');
            plt.show()
            pdb.set_trace()
        '''
            [Q3] Please visualize all the 1st layer's kernels (num=64, 8x8 subplot, average the channel values) using subplot().
            plt.figure(0); plt.imshow(model.features[0].weight.data.clone()[0].mean(dim=0).cpu(), cmap='turbo'); plt.colorbar(); plt.ion(); plt.show()

            <Hints>
            >> How to access kernel: model.features[0].weight.data.clone()
            >> plt.figure(1); plt.imshow(model.features[0].weight.data.clone()[0].mean(dim=0).cpu()); plt.colorbar(); plt.ion(); plt.show()
            >> plt.figure(1); plt.imshow(model.features[0].weight.data.clone()[1].mean(dim=0).cpu()); plt.colorbar(); plt.ion(); plt.show()

            [Q4] Please visualize/compare/analyze the plots: 
            "without pretrained (epoch 0)" vs. "without pretrained (epoch 5)" vs. "with pretrained (epoch 0)" vs. "with pretrained (epoch 5)"
            
            [Q5] Please visualize each layer's activation (conv1~5). Please compare the activations between "without" and "with" pretrained models.
            <Hints>
            >> How to access feature: activation['conv1']
        '''
        if epoch == 4 and i == 99:  
            for cc in range(5):
                plt.figure(figsize=(8, 8)) 
                for k in range(25):
                    plt.subplot(5,5,k+1); 
                    plt.imshow(activation['conv{}'.format(cc+1)][0][k].cpu(), cmap='turbo');
                plt.show()
                pdb.set_trace()

        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Compute gradients
        loss.backward()

        # Update model        
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:         # output loss for every 10 iter
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))   # average loss for every 10 iter
            running_loss = 0.0  # init running_loss

print('Finished Training')



##############################################
################# Test Phase #################

correct = 0
total = 0
with torch.no_grad():   # no_grad for the evalaution step
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

pdb.set_trace()
'''
    [Q6] Please compare the performance between "without" and "with" pretrained models. Why do they have different results?
    
    # With pretrained models (lowest loss : [5, 60] 0.275)
    Accuracy of the network on the 10000 test images: 79 %
    Accuracy for class plane is: 75.9 %
    Accuracy for class car   is: 78.0 %
    Accuracy for class bird  is: 63.7 %
    Accuracy for class cat   is: 72.8 %
    Accuracy for class deer  is: 74.4 %
    Accuracy for class dog   is: 77.8 %
    Accuracy for class frog  is: 89.0 %
    Accuracy for class horse is: 87.2 %
    Accuracy for class ship  is: 90.9 %
    Accuracy for class truck is: 88.9 %

    # Without pretrained models (lowest loss : [5, 60] loss: 0.747)
    Accuracy of the network on the 10000 test images: 66 %
    Accuracy for class plane is: 69.5 %
    Accuracy for class car   is: 75.7 %
    Accuracy for class bird  is: 53.3 %
    Accuracy for class cat   is: 55.3 %
    Accuracy for class deer  is: 51.9 %
    Accuracy for class dog   is: 50.1 %
    Accuracy for class frog  is: 79.0 %
    Accuracy for class horse is: 80.3 %
    Accuracy for class ship  is: 77.4 %
    Accuracy for class truck is: 77.4 %

'''

