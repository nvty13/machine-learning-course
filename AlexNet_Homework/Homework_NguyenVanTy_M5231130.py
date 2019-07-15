#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import math
import os
import matplotlib.pyplot as plt
import copy


# In[5]:


def imshow(original, mean, std):
    img = copy.deepcopy(original)
    for channel in range(len(img)):
        img[channel] = img[channel] * std[channel] + mean[channel]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# In[6]:


pretrain_alexnet = models.alexnet(pretrained=True)


# In[7]:


pretrain_alexnet


# In[8]:


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('')
    print('output:\n', output.data)
    name_of_each_layer.append(self.__class__.__name__)
    output_size_of_each_layer.append(output.data.size())
    outputs_of_each_layer.append(output.data)


# In[9]:


def printclassifier(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('')
    print('output:\n', output.data)
    name_of_each_layer_classifier.append(self.__class__.__name__)
    output_size_of_each_layer_classifier.append(output.data.size())
    outputs_of_each_layer_classifier.append(output.data)


# In[10]:


name_of_each_layer = []
output_size_of_each_layer = []
outputs_of_each_layer = []


# In[11]:


name_of_each_layer_classifier = []
output_size_of_each_layer_classifier = []
outputs_of_each_layer_classifier = []


# In[12]:


for i in range(len(pretrain_alexnet.features)):
    pretrain_alexnet.features[i].register_forward_hook(printnorm)


# In[13]:


for i in range(len(pretrain_alexnet.classifier)):
    pretrain_alexnet.classifier[i].register_forward_hook(printclassifier)


# In[14]:


image_fp = "BEN.jpg"
img_PIL = Image.open(image_fp)
img_PIL


# In[16]:


# 前処理
preprocess1 = transforms.Compose([
   transforms.Scale(224),
   transforms.CenterCrop(224)  # 画像を正方形で切り取る
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess2 = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize(
       mean=mean,
       std=std
   )
])


# In[17]:


img_center_crop = preprocess1(img_PIL)
img_tensor = preprocess2(img_center_crop)
img_tensor.unsqueeze_(0)


# In[18]:


save_dir = os.path.join(os.getcwd(), os.path.splitext(image_fp)[0])
os.makedirs(save_dir, exist_ok=True)


# In[19]:


img_center_crop


# In[20]:


img_center_crop.save(os.path.join(save_dir, 'input_image.jpg'), 'JPEG', quality=100, optimize=True)


# In[21]:


imshow(img_tensor[0], mean, std)
plt.show()


# In[22]:


pretrain_alexnet.eval()
output = pretrain_alexnet.forward(img_tensor)
softmax = nn.Softmax()
output = softmax(output)
print("last output:\n", output)


# In[23]:


values, indices = output.max(1)
print(indices)
print("%f"%values)


# In[25]:


class_index = json.load(open('imagenet_class_index.json', 'r'))


# In[26]:


print(np.max(output.data.numpy()), class_index[str(np.argmax(output.data.numpy()))])


# In[27]:


outputs_of_each_layer


# In[29]:


def plot_out(images, title=None, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    for i in range(len(images)):
        plt.subplot(math.ceil(math.sqrt(len(images))), math.ceil(math.sqrt(len(images))), i+1)
        plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
        plt.tick_params(labelleft="off",left="off") # y軸の削除
        plt.box("off") #枠線の削除
        plt.imshow(images[i])


# In[30]:


for i, (name, output_size, each_layer_output) in enumerate(zip(name_of_each_layer, output_size_of_each_layer, outputs_of_each_layer)):
    title = str(i) + "\tName:" + name+"\tOutput size:"+str(list(output_size[1:]))
    print(title)
    plt.gray()
    plot_out(each_layer_output.data[0], title=title)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'features_' + str(i) + '.png'))
    plt.show()


# In[31]:


for i, (name, output_size, each_layer_output) in enumerate(zip(name_of_each_layer_classifier, output_size_of_each_layer_classifier, outputs_of_each_layer_classifier)):
    title = str(i) + "\tName:" + name+"\tOutput size:"+str(list(output_size[1:]))
    print(title)
    plt.gray()
    plt.imshow(each_layer_output.data, extent=(0,len(each_layer_output[0].data),0,len(each_layer_output[0].data)/10))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'classifier_' + str(i) + '.png'))
    plt.show()


# In[ ]:




