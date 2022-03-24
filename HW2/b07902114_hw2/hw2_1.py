"""## Preparing Data
Load the training and testing data from the `.npy` file (NumPy array).
"""

config = {
    "epochs" : 500,
    "batch_size" : 256,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
        # 'momentum': 0.9,
        # 'weight_decay':0.001,
        'amsgrad':True
    },
}


import numpy as np

def norm(data, mean_data, std_data):
    data = (data - mean_data) / std_data
    return data

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
# train_mean = np.mean(train, axis = 0)
# train_std = np.std(train, axis = 0)
# train = norm(train, train_mean, train_std)
train = train.reshape([len(train),11, 39])
train_label = np.load(data_root + 'train_label_11.npy')

# test = np.load(data_root + 'test_11.npy')
# test = test.reshape([len(test), 11, 39])

# test = norm(test, train_mean, train_std)

print('Size of training data: {}'.format(train.shape))
# print('Size of testing data: {}'.format(test.shape))

"""## Create Dataset"""

import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        self.data = self.data.permute([0,2,1])
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

"""Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data."""

VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
# train_indices = [i for i in range(len(train)) if i % 10 != 1]
# val_indices = [i for i in range(len(train)) if i % 10 == 1]

# train_x, train_y, val_x, val_y = train[train_indices], train_label[train_indices], train[val_indices], train_label[val_indices]
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
# train_x, train_y, val_x, val_y = train[percent:], train_label[percent:], train[:percent], train_label[:percent]

print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

"""Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here."""

BATCH_SIZE = config["batch_size"]

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

"""Cleanup the unneeded variables to save memory.<br>

**notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**
"""

import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

"""## Create Model

Define model architecture, you are encouraged to change and experiment with the model architecture.
"""

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.layer1 = nn.Linear(429, 1024)
        # self.layer2 = nn.Linear(1024, 512)
        # self.layer3 = nn.Linear(512, 128)
        # self.out = nn.Linear(128, 39) 
        # self.act_fn = nn.Sigmoid()
        self.cnn = nn.Sequential(
            nn.Conv1d(39, 256, 3, padding=1, padding_mode='replicate'), #[batch, 256, 11]
            nn.BatchNorm1d(256),
            nn.Dropout(0.7),
            nn.ReLU(),

            nn.Conv1d(256, 512, 3, padding=1, padding_mode='replicate'), #[batch, 128, 5]
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.ReLU(),


            nn.Conv1d(512, 1024, 3, padding=1, padding_mode='replicate'), #[batch, 256, 5]
            nn.BatchNorm1d(1024),
            nn.Dropout(0.7),
            nn.ReLU(),
            # nn.AvgPool1d(2)                          
            
            # nn.MaxPool1d(2), #[batch, 256, 5]
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 * 11, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(128, 39),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

"""## Training"""

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

"""Fix random seeds for reproducibility."""

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""Feel free to change the training parameters here."""

# fix random seed for reproducibility
same_seeds(101)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = config["epochs"]               # number of training epoch
# learning_rate = config["lr"]      # learning rate

# the path where checkpoint saved
model_path = './model_1.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
print(config)
print(model)
# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 
        Acc = (train_pred.cpu() == labels.cpu()).sum().item()
        print('Train Train Acc: {:3.6f} Loss: {:3.6f}'.format(Acc/len(labels), batch_loss.item()), end='\r')
        train_acc += Acc
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

