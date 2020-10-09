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
import cv2

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


def l2_BCE(t, y, eps=1e-12):
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

def store_model(G=False,D=False, folder_name=1, drive=True):
    folder_name = str(folder_name)
    base_paths = ["/content/model_outputs/"]
    if(drive):
      base_paths.append("/content/gdrive/My Drive/Colab Notebooks/LJMU/Custom Code/models/")
    
    for base_path in base_paths:
      os.makedirs(base_path, exist_ok=True)
      model_path = base_path + folder_name + "/"
      try: shutil.rmtree(model_path)
      except: pass
      os.mkdir(model_path)
      if(G):
        torch.save(G.state_dict(), model_path+"G.pth")
      if(D):
        torch.save(D.state_dict(), model_path+"D.pth")

def denoising_epoch_print(x_real, x_noise, G=False,D=False):

    print_queue=[
              tanh2sigmoid(torch.unbind(x_real, dim=2)[0][0]), 
              tanh2sigmoid(torch.unbind(x_noise, dim=2)[0][0]),   
    ]

    if(G):
      print_queue+=[
            tanh2sigmoid(torch.unbind(G(x_real).detach(), dim=2)[0][0]), 
            tanh2sigmoid(torch.unbind(G(x_noise).detach(), dim=2)[0][0]),
            torch.abs(torch.unbind(x_real-G(x_real), dim=2)[0][0]), 
            torch.abs(torch.unbind(x_noise-G(x_noise), dim=2)[0][0]),
      ]

    if(D):
      print_queue+=[
            torch.unbind(D(x_real).detach(), dim=2)[0][0][0],
            torch.unbind(D(x_noise).detach(), dim=2)[0][0][0]
      ]
    
    while(print_queue):
      print_length = min(4,len(print_queue))
      view_img(print_queue[:print_length])
      print_queue = print_queue[print_length:]

def epoch_print(input_img, target, print_result, epoch, G=False, D=False):

    print(print_result)

    with open("../log.txt", "a") as fp:
      fp.write(print_result+"\n")

    if(epoch%10==9):
      try:
        store_model(G, D, drive=True)
      except:
        pass
    else:
      store_model(G, D, drive=False)

    print_queue=[
              tanh2sigmoid(torch.unbind(target, dim=2)[0][0])   
    ]

    if(G):
      print_queue+=[
            tanh2sigmoid(torch.unbind(G(input_img).detach(), dim=2)[0][0]), 
            torch.abs(torch.unbind(target-G(input_img), dim=2)[0][0]), 
      ]

    if(D):
      print_queue+=[
            torch.unbind(D(input_img).detach(), dim=2)[0][0][0]
      ]
    
    while(print_queue):
      print_length = min(4,len(print_queue))
      view_img(print_queue[:print_length])
      print_queue = print_queue[print_length:]

def get_video_eval_frames(original, mask, predicted, text=None):
    original = Image.fromarray(original)
    mask = Image.fromarray(mask)
    original_masked = Image.blend(original,mask, alpha=0.5)
    np_original = np.array(original)
    np_predicted = np.array(predicted)
    np_original_masked = np.array(original_masked)

    if(text):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(np_original,  
                    str(text["file"]),  
                    (15, 15), font, 0.5, (0, 255, 255), 1,  
                    cv2.LINE_4)

        cv2.putText(np_original,  
                    "Original",  
                    (15, 40), font, 0.5, (0, 255, 255), 1,  
                    cv2.LINE_4)

        cv2.putText(np_original_masked,  
                    "Masked", 
                    (15, 20), font, 0.5, (0, 255, 255), 1,  
                    cv2.LINE_4)

        
        cv2.putText(np_original_masked,  
                    "{}".format(text["anomaly_score"]), 
                    (15, 35), font, 0.3, (0, 255, 255), 1,  
                    cv2.LINE_4)

        if(text["auc"]):
          cv2.putText(np_original_masked,  
                      "AUC: {}".format(np.round(text["auc"], 2)), 
                      (15, 60), font, 0.4, (0, 255, 255), 1,  
                      cv2.LINE_4)

    final_image = np.concatenate((np_original, np_original_masked, np_predicted), axis=1)
    return final_image


def get_video_train_frames(original, predicted, text=None):
    if(text):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(original,  
                    str(text),  
                    (15, 15), font, 0.5, (0, 255, 255), 1,  
                    cv2.LINE_4)

    final_image = np.concatenate((original, predicted), axis=1)
    return final_image



def write_video(frame_list, video_name="output.mp4", drive=False):
    video_names = ["/content/"+video_name]
    if(drive):
      video_names.append("/content/gdrive/My Drive/Colab Notebooks/LJMU/Custom Code/videos/"+video_name)
    
    for video_name in video_names:
      writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"XVID"), 10,(frame_list[0].shape[1],frame_list[0].shape[0]))
      for frame in frame_list:
          writer.write(frame.astype('uint8'))
      writer.release()



def combine_eval_frame(row, auc=None):
  score_map = [key for key in row.keys() if "_nor" in key]
  combined =  get_video_eval_frames(
        row["x_real"],
        row["x_mask"],
        row["x_pred"],
        text = {
            "file": row["file_name"],
            "anomaly_score": ", ".join([key.split("_")[0]+":"+str(np.round(row[key],2)) for key in score_map]),
            "auc":auc
        })
  return combined