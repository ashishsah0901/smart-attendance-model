import torch
import torchvision
import config
import torch.optim as optim
import torch.nn.functional as F
from dataset import SiameseDataset
from model import SiameseNetwork
from torch.utils.data import DataLoader
from utils import load_checkpoint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch

model = SiameseNetwork().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(),lr=config.LR)
load_checkpoint("my_checkpoint.pth.tar",model,optimizer,config.LR)
val_dataset = SiameseDataset(root_dir=config.VAL_DIR,transform=config.TRANS,should_invert=False)
# print(type(val_dataset))
test_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
dataiter = iter(test_loader)

image3 = Image.open('image3.png')
image3 = ImageOps.grayscale(image3)
image3 = image3.convert('L')
image3 = config.TRANS(image3)
image3 = image3.unsqueeze(0)
image4 = Image.open('image4.png')
image4 = ImageOps.grayscale(image4)
image4 = image4.convert('L')
image4 = config.TRANS(image4)
image4 = image4.unsqueeze(0)
image5 = Image.open('image5.png')
image5 = ImageOps.grayscale(image5)
image5 = image5.convert('L')
image5 = config.TRANS(image5)
image5 = image5.unsqueeze(0)
output1,output2 = model(image3.to(config.DEVICE),image4.to(config.DEVICE))
euclidean_distance = F.pairwise_distance(output1, output2)
print(euclidean_distance)
# print(type(image0))
# x0,_,_ = next(dataiter)
# # print(type(x0))
# def imshow(img,text=None):
#     npimg = img.numpy()
#     plt.axis("off")
#     if text:
#         plt.text(75, 8, text, style='italic',fontweight='bold',
#             bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.savefig('output.png') 

# # for i in range(1):
# _,x1,label2 = next(dataiter)
# concatenated = torch.cat((x0,x1),0)

# output1,output2 = model(x0.to(config.DEVICE),x1.to(config.DEVICE))
# euclidean_distance = F.pairwise_distance(output1, output2)
# imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))