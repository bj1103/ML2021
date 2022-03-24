# -*- coding: utf-8 -*-
"""
Refernece : https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW03/HW03.ipynb
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torchvision.models import vgg16_bn
from tqdm.auto import tqdm
import random

config = {
    "batch" : 32,
    "lr" : 1e-5,
    "wd" : 1e-3,
    "epochs" : 500,
}


transform_set = [ 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-75, 75)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.3, interpolation=2),
    transforms.AutoAugment(),
    transforms.ColorJitter(brightness=(1, 2), contrast=(1, 2)),
]
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply(transform_set, p=0.7),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


batch_size = config["batch"]
prepath = "/tmp2/b07902114/ML/HW3/"
# prepath = "./drive/My Drive/ML/HW3/"
# prepath = "./"
train_set = DatasetFolder(prepath+"food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(prepath+"food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(prepath+"food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(prepath+"food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
teacher_set = DatasetFolder(prepath+"food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),  
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
            # nn.Dropout(p=0.3),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),

        )
        self.avg = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(4096, 11, bias=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.avg(out)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),  
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
            nn.Dropout(p=0.3),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.6),

        )
        self.avg = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(4096, 11, bias=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.avg(out)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def get_pseudo_labels(dataset, model, threshold=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(teacher_set, batch_size=batch_size, shuffle=False)

    softmax = nn.Softmax(dim=-1)

    total_probs = []
    pseudo_labels = []
    for batch in tqdm(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)
        
        max_prob, max_indices = torch.max(probs, 1)
        if len(total_probs) == 0:
            total_probs = max_prob
            pseudo_labels = max_indices
        else:
            total_probs = torch.cat((total_probs, max_prob))
            pseudo_labels = torch.cat((pseudo_labels, max_indices))
    
    select = [i for i in range(len(total_probs)) if total_probs[i] > threshold]
    dataset.samples = [(d, pseudo_labels[i].cpu().item()) for i, (d, l) in enumerate(dataset.samples)]
    dataset = Subset(dataset, select)
    
    return dataset

device = "cuda" if torch.cuda.is_available() else "cpu"



teacher = Teacher().to(device)
teacher.device = device
teacher.load_state_dict(torch.load(prepath+"model_2.ckpt"))
teacher.eval()
pseudo_set = get_pseudo_labels(unlabeled_set, teacher, 0.9)
concat_dataset = ConcatDataset([train_set, pseudo_set])
train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
del teacher


model = Classifier().to(device)
model.device = device
# model.load_state_dict(torch.load(prepath+"model.ckpt"))
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"], amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["wd"], momentum=0.9, nesterov=True)

n_epochs = config["epochs"]



do_semi = False
max_val_acc = 0.4
softmax = nn.Softmax(dim=-1)
for epoch in range(n_epochs):
    if do_semi:
        pseudo_set = get_pseudo_labels(unlabeled_set, model, 0.92 - (0.1) * (epoch / n_epochs))
        concat_dataset = ConcatDataset([train_set, pseudo_set, pseudo_set_2])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
   
    model.train()

    train_loss = []
    train_accs = []

    # for batch, batch_un in zip(train_loader, unlabeled_set):
    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
    model.eval()
    valid_loss = []
    valid_accs = []
    for batch in valid_loader:
        imgs, labels = batch
        with torch.no_grad():
          logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    if valid_acc > max_val_acc:
        torch.save(model.state_dict(), prepath+"model_4.ckpt")
        max_val_acc = valid_acc
        print(f"Saving model with val acc = {valid_acc:.5f}")


del model

model = Classifier().to(device)
model.device = device

model.load_state_dict(torch.load(prepath+"model_4.ckpt"))
model.eval()

predictions = []

for batch in tqdm(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

with open(prepath+"predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
