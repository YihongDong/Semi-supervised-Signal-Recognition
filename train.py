#encoding:utf-8
import numpy as np
from SSRCNN import getSSRCNN
from dataset import DataSet
from sklearn.preprocessing import MinMaxScaler
from center_loss import CenterLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
from torch.autograd import Variable
from kl_divergence import kl_categorical 

# hyperparameters
num_class = 11
batchsize = 64
num_channels = 2
lr = 1e-3
maxprecision = 0
maxs = []
max_epoch_num = 350
lam1 = 1
lam2 = 0.003
result_str=''

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataC = DataSet('/home/dongyihong/data/2016.04C.multisnr.pkl')
X_train_label, Y_train_label, X_train_unlabel, Y_train_unlabel, X_test, Y_test = dataC.getTrainAndTest()

X_train = X_train_label.reshape((-1, num_channels, 256//num_channels, 1))
X_train_unlabel = X_train_unlabel.reshape((-1, num_channels, 256//num_channels, 1))
X_test = X_test.reshape((-1, num_channels, 256//num_channels, 1))

# data loader
train_dataset = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train_label))
train_loader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
test_dataset = Data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
test_loader = Data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=2)
train_unlabel_dataset = Data.TensorDataset(torch.Tensor(X_train_unlabel), torch.Tensor(Y_train_unlabel))
train_unlabel_loader = Data.DataLoader(train_unlabel_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

unlabeled_dataiter = iter(train_unlabel_loader)

# loss func
criterion_cross = nn.CrossEntropyLoss()
criterion_unlabeled_cross = nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=num_class, feat_dim=100, use_gpu=torch.cuda.is_available())
criterion_kl = kl_categorical

# model
model = getSSRCNN(num_classes=num_class,num_channels=num_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer_cent = optim.Adam(criterion_cent.parameters(), lr=lr)

print(model)

# add noise for kl loss computation
def addNoise(unlabelInput):
    d = np.random.randn(unlabelInput.shape[0], num_channels, 256//num_channels, 1)
    Noise = [np.sqrt(np.sum(abs(unlabelInput[i].numpy())**2)/(50*np.sum(abs(d[i])**2)))*d[i] \
                     for i in range(unlabelInput.shape[0])]
    unlabelNoiseInput = unlabelInput + torch.Tensor(np.array(Noise))
    return unlabelNoiseInput

# train
for epoch in range(max_epoch_num):
    running_loss = 0.0
    running_cross = 0.0
    running_unlabeled_cross = 0.0
    running_cent = 0.0
    running_kl = 0.0
    count=0
    for i, data in enumerate(train_loader):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        optimizer_cent.zero_grad()

        outputs = model(inputs)

        loss_cross = criterion_cross(outputs, labels.long())
        loss_cent = criterion_cent(model.getSemantic(inputs), labels.long())

        unlabelInput, unlabels = next(unlabeled_dataiter)
        unlabelInput = unlabelInput.cpu()
        unlabelNoiseInput = addNoise(unlabelInput).to(device)
        unlabelInput = unlabelInput.to(device)
        unlabelOutput = model(unlabelInput)

        loss_kl = torch.zeros(1).to(device)
        loss_unlabeled_cross = torch.zeros(1).to(device)

        unlabelNoiseOutput = model(unlabelNoiseInput)
        loss_kl+=criterion_kl(unlabelOutput,unlabelNoiseOutput)

        if epoch > 50:
            _, fake_label = torch.max(unlabelOutput, 1)
            loss_unlabeled_cross += criterion_unlabeled_cross(unlabelOutput, fake_label.long())

        if callable(unlabeled_dataiter) == False:
            unlabeled_dataiter = iter(train_unlabel_loader)

        print('cross: {:.4f}, auto: {:.4f}, cent: {:.4f}, kl: {:.4f}'.format(loss_cross.item(), loss_unlabeled_cross.item(), loss_cent.item(), loss_kl.item()))

        loss = loss_cross + loss_unlabeled_cross + lam1 * loss_kl + lam2 * loss_cent
        loss.backward()
        optimizer.step()
        optimizer_cent.step()

        count+=inputs.shape[0]

        running_loss += loss.item() * inputs.shape[0]
        running_cross += loss_cross.item() * inputs.shape[0]
        running_unlabeled_cross += loss_unlabeled_cross.item() * inputs.shape[0]
        running_cent += loss_cent.item() * inputs.shape[0]
        running_kl += loss_kl.item() * inputs.shape[0]

    print('[%d , %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))

    class_correct = torch.ones(num_class).to(device)
    class_total = torch.ones(num_class).to(device)
    for data in test_loader:
        model.eval()
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.to(device)).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[int(label)] += c[i]
            class_total[int(label)] += 1

    precision = 0
    temp = []
    for i in range(num_class):
        temp.append('Accuracy: %.2f %% %d' % (100 * class_correct[i] / class_total[i], class_total[i]))
        precision += 100 * class_correct[i] / class_total[i]
    print('Total accuracy: %.2f %%' % (100 * class_correct.sum()/class_total.sum()))
    
    if class_correct.sum()/class_total.sum() > maxprecision:
        maxs= temp
    maxprecision =max(class_correct.sum()/class_total.sum(), maxprecision)
    running_loss = 0.0

    result_str+='{}, {:.2f}, {:.4f}, {:.4f}, {:.4f}\n'.format(epoch, class_correct.sum()/class_total.sum()*100, running_cross/count, running_unlabeled_cross/count, running_cent/count, running_kl/count)


torch.save(model, './model.pkl')
print('Finished Training')

# test
class_correct = torch.ones(num_class).to(device)
class_total = torch.ones(num_class).to(device)
for data in test_loader:
    images, labels = data
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.to(device)).squeeze()
    for i in range(len(labels)):
        label = labels[i]
        class_correct[int(label)] += c[i]
        class_total[int(label)] += 1

torch.save({'model': model.state_dict()}, './model.pkl')

precision = 0
for i in range(num_class):
    print('Accuracy: %2d %%' % (100 * class_correct[i] / class_total[i]))
    precision += 100 * class_correct[i] / class_total[i]
print('Total accuracy: %2d %%' % (100 * class_correct.sum()/class_total.sum()))

print('Max total accuracy: %.2f %%' % (100 * maxprecision))
for i in maxs:
    print(i)

with open('./result.txt','w') as f:
     f.write(result_str)
