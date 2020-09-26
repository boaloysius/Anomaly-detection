import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import shutil
import os
import sys

def convert_tensor_to_PIL(tensor, out_size=None):
    out = transforms.ToPILImage()(tensor.cpu())
    if out_size is not None:
        out = out.resize(out_size)
    return out


def show_imgs(imgs, idxs=[0], save_path=None):
    n = len(imgs)
    
    for idx in idxs:
        plt.figure(figsize=(20, 5))
        for i, (k, v) in enumerate(imgs.items()):
            ax = plt.subplot(1, n, i+1)
            if k[0] == 'x':
                ax.imshow(convert_tensor_to_PIL(v[idx]), cmap='gray')
            else:
                sns.heatmap(
                    v[idx][0].data.cpu().numpy(), vmin=0, vmax=1, 
                    cmap='gray', square=True, cbar=False)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])    
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        
        
def tanh2sigmoid(x):
    return (x + 1) / 2 


def torch_log(x, eps=1e-12):
    return torch.log(torch.clamp(x, eps, 1.))


def l2_BCE(y, t, eps=1e-12):
    return -(t*torch_log(y**2) + (1-t)*torch_log((1-y)**2)).mean()


def get_optimizer(optimizer, params, lr, momentum=0.9):
    if optimizer == 'Adam':
        return optim.Adam(params, lr=lr)
    else:
        return optim.SGD(params, lr=lr, momentum=momentum)
    
    
# for test
def draw_real_gt(ax, real_img, gt_img):
    ax.imshow(real_img, cmap='gray')
    ax.imshow(gt_img, alpha=0.5, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('x_real + gt')
    

def draw_fake(ax, x_fake):
    ano_img = transforms.ToPILImage()(tanh2sigmoid(x_fake).cpu()[0])
    ax.imshow(ano_img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('x_fake')


def draw_heatmap(ax, x, title='', cmap='gray'): 
    sns.heatmap(x, vmin=0, vmax=1, cmap=cmap, ax=ax, square=True, cbar=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    
def visualize(real_img, gt_img, x_real, x_fake=None, D_real=None, save_path=None):
    
    plt.figure(figsize=(20,15))

    # real + gt
    ax = plt.subplot(3, 4, 1)
    draw_real_gt(ax, real_img, gt_img)

    #x_real
    ax = plt.subplot(3, 4, 2)
    x_real=np.transpose(x_real,(1,2,0))
    ax.imshow(x_real)

    #x_fake
    if(x_fake is not None):
      ax = plt.subplot(3, 4, 3)
      x_fake = np.transpose(x_fake,(1,2,0))
      ax.imshow(x_fake)

    #D_real
    if(D_real is not None):
      ax = plt.subplot(3, 4, 3)
      D_real = D_real[0]#np.transpose(D_real,(1,2,0))
      sns.heatmap(D_real, vmin=0, vmax=1,cmap='gray', square=True, cbar=False)
      #ax.imshow(D_real)

    plt.show()
    return
    # fake
    ax = plt.subplot(3, 4, 2)
    draw_fake(ax, x_fake)

    # D_fake
    ax = plt.subplot(3, 4, 3)
    draw_heatmap(ax, _D_fake, 'D_fake')

    # heat
    ax = plt.subplot(3, 4, 4)
    draw_heatmap(ax, heat, '|I(X~) - X|')

    # |I(X~)-X| threshold
    for a, alpha in enumerate(alphas):
        ax = plt.subplot(3, 4, 5+a)
        draw_heatmap(ax, heat>alpha, 'alpha = {:.1f}'.format(alpha))
   
    # D_fake threshold
    for z, zeta in enumerate(zetas):
        ax = plt.subplot(3, 4, 9+z)
        draw_heatmap(ax, _D_fake<zeta, 'zeta = {:.1f}'.format(zeta))

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def view_img(imgs, title=None, save_path=None, heat_index=[]):
    import torchvision.transforms as transforms
    import seaborn as sns

    fig, axs = plt.subplots(1, len(imgs))
    fig.set_size_inches(10, 10)
    for i, img in enumerate(imgs):
      if(i in heat_index):
        sns.heatmap(img, vmin=0, vmax=1, cmap='gray', square=True, cbar=False)
      elif(type(img) not in [torch.Tensor,np.ndarray]):
        axs[i].imshow(img)
      elif(len(img.shape)==3 and img.shape[2]!=3):
        # To view using imshow. The image should be (x,y,3) shape. 
        # Tensor output by model will be (3,x,y)
        axs[i].imshow(transforms.ToPILImage()(img.cpu()))
      else:
        axs[i].imshow(img)
    if(title):
      fig.suptitle(title, fontsize=30, va="center")
    plt.show()

def tanh2sigmoid(x):
    return (x + 1) / 2 

def store_model(mode, G=False,D=False, name=1):
    name = str(name)
    base_colab_model_path = "../model_outputs/"
    base_gdrive_model_path = "../gdrive/My Drive/Colab Notebooks/LJMU/Custom Code/models/"

    colab_model_path = base_colab_model_path + name + "/"
    gdrive_model_path = base_gdrive_model_path +name + "/"

    shutil.rmtree(colab_model_path)
    shutil.rmtree(gdrive_model_path)

    if("G" in mode):
      torch.save(G.state_dict(), colab_model_path+"G.pth")
      torch.save(G.state_dict(), gdrive_model_path+"G.pt")
    if("D" in mode):
      torch.save(D.state_dict(), colab_model_path+"D.pth")
      torch.save(D.state_dict(), gdrive_model_path+"D.pt")