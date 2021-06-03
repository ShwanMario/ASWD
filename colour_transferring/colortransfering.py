import numpy as np
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte
from skimage.transform import resize
import ot

from sklearn import cluster

from tqdm import tqdm

import sys
import torch

import time
from utils import *


np.random.seed(1)
torch.manual_seed(1)
n_clusters = 3000
name1='1_source.bmp'#path to images 1
name2='1_target.bmp'#path to images 2
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
#MODE=SW,SSW,MSSW,PSSW,MPSSW,MaxSW,DSW
n1=10
n2=100
mode='combo2'
#
if(mode=='cluster'):
    X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_

    # create an array from labels and values
    #source_compressed = np.choose(labels, values)
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape

    vmin = source.min()
    vmax = source.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Source")
    plt.imshow(source,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Source")
    plt.imshow(source_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)
    with open('npzfiles/'+name1+'source_compressed.npy', 'wb') as f:
        np.save(f, source_compressed)
    with open('npzfiles/'+name1+'source_values.npy', 'wb') as f:
        np.save(f, source_values)
    with open('npzfiles/'+name1+'source_labels.npy', 'wb') as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_

    # create an array from labels and values
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape

    vmin = target.min()
    vmax = target.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Target")
    plt.imshow(target,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Target")
    plt.imshow(target_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)

    with open('npzfiles/'+name2+'target_compressed.npy', 'wb') as f:
        np.save(f, target_compressed)
    with open('npzfiles/'+name2+'target_values.npy', 'wb') as f:
        np.save(f, target_values)
    with open('npzfiles/'+name2+'target_labels.npy', 'wb') as f:
        np.save(f, target_labels)
else:
    with open('npzfiles/'+name1+'source_compressed.npy', 'rb') as f:
        source_compressed = np.load(f)
    with open('npzfiles/'+name2+'target_compressed.npy', 'rb') as f:
        target_compressed = np.load(f)
    with open('npzfiles/'+name1+'source_values.npy', 'rb') as f:
        source_values = np.load(f)
    with open('npzfiles/'+name2+'target_values.npy', 'rb') as f:
        target_values = np.load(f)
    with open('npzfiles/'+name1+'source_labels.npy', 'rb') as f:
        source_labels = np.load(f)

if(mode=='SW'):
    f, ax = plt.subplots(1, 4, figsize=(20, 5))
    print(source_values.shape)
    print(target_values.shape)
    print(source.shape)
    ax[0].imshow(source)
    ax[1].imshow(transform_SW(source_values,target_values,source_labels,source,n=n1))
    ax[2].imshow(transform_SW(source_values,target_values,source_labels,source,n=n2))
    ax[3].imshow(reshaped_target)

    ax[0].set_title('Source', fontsize=20)
    ax[1].set_title('SW n='+str(n1), fontsize=20)
    ax[2].set_title('SW n='+str(n2), fontsize=20)
    ax[3].set_title('Target', fontsize=20)

    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/SW_transfer_'+name1+'_to_'+name2+'.pdf')
    plt.show()
#MAX
elif(mode=='MaxSW'):
    f, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(source)
    ax[1].imshow(transform_maxSW(source_values,target_values,source_labels,source))
    ax[2].imshow(reshaped_target)

    ax[0].set_title('Source', fontsize=20)
    ax[1].set_title('Max-SW', fontsize=20)
    ax[2].set_title('Target', fontsize=20)

    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/MaxSW_transfer_'+name1+'_to_'+name2+'.pdf')
    plt.show()
#DSW

elif(mode=='DSW'):
    f, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(source)
    ax[1].imshow(transform_DSW(source_values,target_values,source_labels,source,n=n1))
    ax[2].imshow(transform_DSW(source_values,target_values,source_labels,source,n=n2))
    ax[3].imshow(reshaped_target)

    ax[0].set_title('Source', fontsize=20)
    ax[1].set_title('DSW n='+str(n1), fontsize=20)
    ax[2].set_title('DSW n='+str(n2), fontsize=20)
    ax[3].set_title('Target', fontsize=20)

    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/DSW_transfer_'+name1+'_to_'+name2+'.pdf')
    plt.show()
elif(mode=='combo1'):
    f, ax = plt.subplots(1, 2, figsize=(25,4.5))
    ax[0].imshow(source)
    ax[1].imshow(reshaped_target)
    ax[0].set_title('Source', fontsize=20)
    ax[1].set_title('Target', fontsize=20)
    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/'+name1+'_to_'+name2+'.pdf')
    # plt.show()
    plt.clf()
#########
    f, ax = plt.subplots(1, 5, figsize=(25, 4.5))

    ax[0].imshow(transform_SW(source_values, target_values, source_labels, source, n=n1))
    ax[1].imshow(transform_SW(source_values, target_values, source_labels, source, n=n2))
    ax[2].imshow(transform_maxSW(source_values,target_values,source_labels,source))
    ax[3].imshow(transform_DSW(source_values, target_values, source_labels, source, n=n1))
    ax[4].imshow(transform_DSW(source_values, target_values, source_labels, source, n=n2))

    ax[0].set_title('SW n=' + str(n1), fontsize=20)
    ax[1].set_title('SW n=' + str(n2), fontsize=20)
    ax[2].set_title('MaxSW', fontsize=20)
    ax[3].set_title('DSW n=' + str(n1), fontsize=20)
    ax[4].set_title('DSW n=' + str(n2), fontsize=20)

    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/combo1_' + name1 + '_to_' + name2 + '.pdf')
    plt.show()
elif(mode=='drSW'):
    f, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    ax[0].imshow(source)
    ax[1].imshow(transform_drSW(source_values,target_values,source_labels,source))
    ax[2].imshow(transform_saveSW(source_values, target_values, source_labels, source))
    ax[3].imshow(reshaped_target)

    ax[0].set_title('Source', fontsize=20)
    ax[1].set_title('drSW', fontsize=20)
    ax[2].set_title('saveSW', fontsize=20)
    ax[3].set_title('Target', fontsize=20)

    for axis in ax:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    f.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig('transferimages/drSW_transfer_'+name1+'_to_'+name2+'.pdf')
    plt.show()
