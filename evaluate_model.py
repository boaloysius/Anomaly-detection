import shutil
import os
import sys

from utils import *
import eval_dataset

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

import libs.pytorch_ssim.pytorch_ssim as pytorch_ssim
ssim_metric = pytorch_ssim.ssim

def eval_copy(video_names):
  train_dir = "../data/UCSD_processed/UCSDped1/Train/"
  test_dir  = "../data/UCSD_processed/UCSDped1/Test/"
  gt_dir = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/"
  eval_dir  = "../data/UCSD_processed/UCSDped1/Evaluate/"

  if os.path.exists(eval_dir) and os.path.isdir(eval_dir):
    shutil.rmtree(eval_dir)
  
  for video_name in video_names:
    src    = test_dir + video_name
    src_gt = gt_dir + video_name+"_gt"
    dst    = eval_dir+video_name
    dst_gt = eval_dir+video_name+"_gt"
    shutil.copytree(src, dst)
    shutil.copytree(src_gt, dst_gt)
  
  print(os.listdir(eval_dir))
  sys.exit(0)

def eval_loader():
  eval_dir  = "../data/UCSD_processed/UCSDped1/Evaluate/"
  size = 224
  depth=8
  dataset = eval_dataset.Dataset(eval_dir, (size, size), rgb=True, depth=depth)
  loader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, num_workers=1)
  return loader

def evaluate_temporal_discriminator(model, video_name, kind="Test", threshold=None):
  if(threshold==None):
    threshold=0.999
  eval_copy(video_name, kind)
  loader = eval_loader()

  for index, x_real in enumerate(loader):
    num_frames = x_real.shape[2]
    pred = model(x_real.to(device))
    x_real_frames = x_real.detach().to("cpu").unbind(dim=2)
    #x_real_frames = [x.numpy() for x in x_real_frames]
    pred_frames = pred.detach().to("cpu").unbind(dim=2)
    #pred_frames = [f.numpy() for f in pred_frames]

    for i in range(num_frames):
      anomaly_count = (pred_frames[i] < threshold).sum()
      title = "{} : {} ".format(index*num_frames+i, anomaly_count)
      print(title)
      view_img([tanh2sigmoid(x_real_frames[i][0]), pred_frames[i][0][0]], heat_index=[1])
      #print(i)

def evaluate_generator(model, video_name, kind="Test", threshold=None):
  if(threshold==None):
    threshold=0.999
  eval_copy(video_name, kind)
  loader = eval_loader()

  for index, x_real in enumerate(loader):
    num_frames = x_real.shape[2]
    pred = model(x_real.to(device))
    x_real_frames = x_real.detach().to("cpu").unbind(dim=2)
    #x_real_frames = [x.numpy() for x in x_real_frames]
    pred_frames = pred.detach().to("cpu").unbind(dim=2)
    #pred_frames = [f.numpy() for f in pred_frames]

    for i in range(num_frames):
      anomaly_count = (pred_frames[i] < threshold).sum()
      title = "{} : {} ".format(index*num_frames+i, anomaly_count)
      print(title)
      view_img([
        tanh2sigmoid(x_real_frames[i][0]), 
        tanh2sigmoid(pred_frames[i][0]), 
        torch.abs(x_real_frames[i][0] - pred_frames[i][0])[0]
        ], heat_index=[2])

def evaluate_full_model(G, D, video_name, threshold=None):
  if(threshold==None):
    threshold=0.999
  eval_copy(video_name)
  loader = eval_loader()

  for index, x_real in enumerate(loader):
    num_frames = x_real.shape[2]
    if(G):
      pred_G = G(x_real.to(device))
      pred_frames_G = pred_G.detach().to("cpu").unbind(dim=2)
    if(D):
      pred_D = D(x_real.to(device))
      pred_frames_D = pred_D.detach().to("cpu").unbind(dim=2)
    
    x_real_frames = x_real.detach().to("cpu").unbind(dim=2)
    
    for i in range(num_frames):
      anomaly_count = 1-ssim_metric(pred_frames_G[i],  x_real_frames[i])
      title = "{} : {} ".format(index*num_frames+i, anomaly_count)
      print(title)

      view_list = [tanh2sigmoid(x_real_frames[i][0])]
      if(G):
        view_list.append(tanh2sigmoid(pred_frames_G[i][0]))
      if(D):
        if(len(pred_frames_D)==1):
          view_list.append(pred_frames_D[0][0][0])
        else:
          view_list.append(pred_frames_D[0][0][0])
      if(G):
        view_list.append(torch.abs(x_real_frames[i][0] - pred_frames_G[i][0]))
      view_img(view_list, heat_index=[3])
      #break
    #break