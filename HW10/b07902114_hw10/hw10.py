# -*- coding: utf-8 -*-
"""hw10_adversarial_attack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW10/HW10.ipynb

# **Homework 10 - Adversarial Attack**

Slides: https://reurl.cc/v5kXkk

Videos:

TA: ntu-ml-2021spring-ta@googlegroups.com

## Enviroment & Download

We make use of [pytorchcv](https://pypi.org/project/pytorchcv/) to obtain CIFAR-10 pretrained model, so we need to set up the enviroment first. We also need to download the data (200 images) which we want to attack.
"""

# set up environment
# !pip install pytorchcv

# download
# !gdown --id 1fHi1ko7wr80wXkXpqpqpOxuYH1mClXoX -O data.zip

# unzip
# !unzip ./data.zip
# !rm ./data.zip

"""## Global Settings

* $\epsilon$ is fixed to be 8. But on **Data section**, we will first apply transforms on raw pixel value (0-255 scale) **by ToTensor (to 0-1 scale)** and then **Normalize (subtract mean divide std)**. $\epsilon$ should be set to $\frac{8}{255 * std}$ during attack.

* Explaination (optional)
    * Denote the first pixel of original image as $p$, and the first pixel of adversarial image as $a$.
    * The $\epsilon$ constraints tell us $\left| p-a \right| <= 8$.
    * ToTensor() can be seen as a function where $T(x) = x/255$.
    * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
    * After applying ToTensor() and Normalize() on $p$ and $a$, the constraint becomes $\left| N(T(p))-N(T(a)) \right| = \left| \frac{\frac{p}{255}-mean}{std}-\frac{\frac{a}{255}-mean}{std} \right| = \frac{1}{255 * std} \left| p-a \right| <= \frac{8}{255 * std}.$
    * So, we should set $\epsilon$ to be $\frac{8}{255 * std}$ after ToTensor() and Normalize().
"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8

# the mean and std are the calculated statistics from cifar_10 dataset
cifar_10_mean = (0.491, 0.482, 0.447) # mean for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201) # std for the three channels of cifar_10 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std
# TODO: iterative fgsm attack
# alpha (step size) can be decided by yourself
alpha = 0.8/255/std

root = './data' # directory for storing benign images
# benign images: images which do not contain adversarial perturbations
# adversarial images: images which include adversarial perturbations

"""## Data

Construct dataset and dataloader from root directory. Note that we store the filename of each image for future usage.
"""

import os
import glob
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model
model = ptcv_get_model('resnet56_cifar10', pretrained=True).to(device)
model_2 = ptcv_get_model('nin_cifar10', pretrained=True).to(device)
model_3 = ptcv_get_model('densenet100_k12_cifar10', pretrained=True).to(device)
model_4 = ptcv_get_model('pyramidnet110_a48_cifar10', pretrained=True).to(device)
model_5 = ptcv_get_model('ror3_110_cifar10', pretrained=True).to(device)

model_6 = ptcv_get_model('wrn28_10_cifar10', pretrained=True).to(device)
model_7 = ptcv_get_model('seresnet56_cifar10', pretrained=True).to(device)
model_8 = ptcv_get_model('rir_cifar10', pretrained=True).to(device)
model_9 = ptcv_get_model('resnet164bn_cifar10', pretrained=True).to(device)
model_10 = ptcv_get_model('preresnet164bn_cifar10', pretrained=True).to(device)

model.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()
model_6.eval()
model_7.eval()
model_8.eval()
model_9.eval()
model_10.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')

"""## Utils -- Benign Images Evaluation"""

# to evaluate the performance of model on benign images
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

"""## Utils -- Attack Algorithm"""

# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y) # calculate loss
    loss += loss_fn(model_2(x_adv), y)
    loss += loss_fn(model_3(x_adv), y)
    loss += loss_fn(model_4(x_adv), y)
    loss += loss_fn(model_5(x_adv), y)
    loss += loss_fn(model_6(x_adv), y)
    loss += loss_fn(model_7(x_adv), y)
    loss += loss_fn(model_8(x_adv), y)
    loss += loss_fn(model_9(x_adv), y)
    loss += loss_fn(model_10(x_adv), y)
    # loss += loss_fn(model_11(x_adv), y)

    loss /= 10
    loss.backward() # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    return x_adv

# TODO: perform iterative fgsm attack
# set alpha as the step size in Global Settings section
# alpha and num_iter can be decided by yourself
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=200):
    # initialize x_adv as original benign image x
    x_adv = x.detach().clone()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = fgsm(model, x_adv, y, loss_fn, alpha)
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x+epsilon)
    # for each loop
        # call fgsm with (epsilon = alpha) to obtain new x_adv
        # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

"""## Utils -- Attack

* Recall
    * ToTensor() can be seen as a function where $T(x) = x/255$.
    * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.

* Inverse function
    * Inverse Normalize() can be seen as a function where $N^{-1}(x) = x*std+mean$ where $mean$ and $std$ are constants.
    * Inverse ToTensor() can be seen as a function where $T^{-1}(x) = x*255$.

* Special Noted
    * ToTensor() will also convert the image from shape (height, width, channel) to shape (channel, height, width), so we also need to transpose the shape back to original shape.
    * Since our dataloader samples a batch of data, what we need here is to transpose **(batch_size, channel, height, width)** back to **(batch_size, height, width, channel)** using np.transpose.
"""

# perform adversarial attack and generate adversarial examples
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn) # obtain adversarial examples
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))

"""## Model / Loss Function

Model list is available [here](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). Please select models which has _cifar10 suffix. Some of the models cannot be accessed/loaded. You can safely skip them since TA's model will not use those kinds of models.
"""



# model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

"""## FGSM"""

# adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
# print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

# create_dir(root, 'fgsm', adv_examples, adv_names)

"""## I-FGSM"""

# TODO: iterative fgsm attack
adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

create_dir(root, 'ifgsm', adv_examples, adv_names)

"""## Compress the images"""

# Commented out IPython magic to ensure Python compatibility.
# %cd fgsm
# !tar zcvf ../fgsm.tgz *
# %cd ..

# %cd ifgsm
# !tar zcvf ../ifgsm.tgz *
# %cd ..

# """## Visualization"""

# import matplotlib.pyplot as plt

# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10, 20))
# cnt = 0
# for i, cls_name in enumerate(classes):
#     path = f'{cls_name}/{cls_name}1.png'
#     # benign image
#     cnt += 1
#     plt.subplot(len(classes), 4, cnt)
#     im = Image.open(f'./data/{path}')
#     logit = model(transform(im).unsqueeze(0).to(device))[0]
#     predict = logit.argmax(-1).item()
#     prob = logit.softmax(-1)[predict].item()
#     plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
#     plt.axis('off')
#     plt.imshow(np.array(im))
#     # adversarial image
#     cnt += 1
#     plt.subplot(len(classes), 4, cnt)
#     im = Image.open(f'./fgsm/{path}')
#     logit = model(transform(im).unsqueeze(0).to(device))[0]
#     predict = logit.argmax(-1).item()
#     prob = logit.softmax(-1)[predict].item()
#     plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
#     plt.axis('off')
#     plt.imshow(np.array(im))
# plt.tight_layout()
# plt.show()