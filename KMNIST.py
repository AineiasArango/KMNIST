import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#load data
test_imgs = np.load("C:\\Users\\sakai\\kmnist-test-imgs.npz")
test_lbls = np.load("C:\\Users\sakai\kmnist-test-labels.npz")
train_imgs = np.load("C:\\Users\sakai\kmnist-train-imgs.npz")
train_lbls = np.load("C:\\Users\sakai\kmnist-train-labels.npz")

test_image_list = torch.from_numpy(test_imgs['arr_0']).float()
test_image_list = test_image_list[:,None,:,:]
y_test = torch.from_numpy(test_lbls['arr_0']) #labels

train_image_list = torch.from_numpy(train_imgs['arr_0']).float()
train_image_list = train_image_list[:,None,:,:]
y_train = torch.from_numpy(train_lbls['arr_0']) #labels

#create convolutional neural network
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1)) #same convolution
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1)) #same convolution
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    

#load data to use in batches
class Data(Dataset):
    def __init__(self):
        self.x = train_image_list
        self.y = y_train
        self.len = self.x.shape[0]
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
data_set = Data()

batchsize = 64
learning_rate = 0.001

trainloader = DataLoader(dataset = data_set, batch_size = batchsize)

#create model, optimizer, and loss function
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

#perform training
num_epochs = 1
#5-89.2
loss_list = []

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(trainloader):
        #forward
        z = model(x)
        loss = criterion(z,y)
        #backward
        optimizer.zero_grad()
        loss.backward()
        #gradient descent
        optimizer.step()
        loss_list.append(loss.data)
 
        
#%%

predict = model(test_image_list)
yhat = torch.max(predict,1)      
correct = torch.sum(yhat[1] == y_test).float()
percent_correct = 100*correct.numpy()/len(y_test)
print(percent_correct)



















