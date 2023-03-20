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
test_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
dataiter = iter(test_loader)

# image8 = Image.open('image2.jpeg')
# image8 = ImageOps.grayscale(image8)
# image8 = image8.convert('L')
# image8 = config.TRANS(image8)
# image8 = image8.unsqueeze(0)
# image9 = Image.open('image1.jpg')
# image9 = ImageOps.grayscale(image9)
# image9 = image9.convert('L')
# image9 = config.TRANS(image9)
# image9 = image9.unsqueeze(0)

# output1,output2 = model(image9.to(config.DEVICE),image8.to(config.DEVICE))
# euclidean_distance = F.pairwise_distance(output1, output2)
# print(euclidean_distance)
# print(type(image0))
x0,_,_ = next(dataiter)
# print(type(x0))
def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('output.png') 

for i in range(10):
  _,x1,label2 = next(dataiter)
  concatenated = torch.cat((x0,x1),0)

  output1,output2 = model(x0.to(config.DEVICE),x1.to(config.DEVICE))
  euclidean_distance = F.pairwise_distance(output1, output2)
  imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))