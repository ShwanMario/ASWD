import torch
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
import imageio
import numpy as np
import argparse
from DCGANAE import DCGANAE,Discriminator
import torch
import argparse
import os
from torch import  optim

from torchvision import transforms

from experiments import sampling
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import random
parser = argparse.ArgumentParser(description='Generate fake data')
parser.add_argument('--model-type', type=str, required=True,
                    help='(SWD|MSWD|DSWD|GSWD|DGSWD|ASWD|)')
parser.add_argument('--num-images', type=int, default=20000,
                    help='number of image to generate')
parser.add_argument('--batch-size',type=int,required=True,
                    help='batch size in training')
parser.add_argument('--num-projection',type=int,required=True,
                    help='number of projections')
parser.add_argument('--latent-size',type=int,required=True,
                    help='dimension of latent variable')
parser.add_argument('--dataset',type=str,required=True,help='dataset')
parser.add_argument('--gpu',type=str,required=False,default=0)
args = parser.parse_args()
latent_size=args.latent_size
device='cuda:'+args.gpu

model = DCGANAE(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(device)

model.load_state_dict(torch.load('./result/'+args.dataset+'/'+args.model_type+'_'+str(args.batch_size)+'_'+str(args.num_projection)+'_'+str(latent_size)+'_model.pth'))

fixednoise= torch.randn((args.num_images, latent_size)).to(device)

imgs=model.decoder(fixednoise)
import os
datadir='./result/'+args.dataset+'/fake/'+args.model_type+'_'+str(args.batch_size)+'_'+str(args.num_projection)+'_'+str(latent_size)+'/'
if not (os.path.isdir(datadir)):
    os.mkdir(datadir)

for i,img in enumerate(imgs):
    img=img.transpose(0,-1).transpose(0,1).cpu().detach().numpy()
    img=(img*255).astype(np.uint8)
    imageio.imwrite('./result/'+args.dataset+'/fake/'+args.model_type+'_'+str(args.batch_size)+'_'+str(args.num_projection)+'_'+str(latent_size)+'/'+str(i)+'.png', img)
