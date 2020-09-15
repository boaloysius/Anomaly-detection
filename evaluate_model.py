import shutil
import os

from utils import *
import dataset1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_copy(video_name, kind="Test"):
  train_dir = "../data/UCSD_processed/UCSDped1/Train/"
  test_dir  = "../data/UCSD_processed/UCSDped1/Test/"
  eval_dir  = "../data/UCSD_processed/UCSDped1/Evaluate/"
  src = (train_dir if kind=="Train" else test_dir) + video_name
  dst = eval_dir+video_name

  if os.path.exists(eval_dir) and os.path.isdir(eval_dir):
    shutil.rmtree(eval_dir)
  shutil.copytree(src, dst)
  os.listdir(eval_dir)

def eval_loader():
  eval_dir  = "../data/UCSD_processed/UCSDped1/Evaluate/"
  size = 224
  depth=10
  dataset = dataset1.Dataset(eval_dir, (size, size), rgb=True, depth=depth)
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
      view_img([tanh2sigmoid(x_real_frames[i][0]), x_real_frames[i][0]-pred_frames[i][0][0]], heat_index=[1])

def evaluate_full_model(G, D, video_name, kind="Test", threshold=None):
  if(threshold==None):
    threshold=0.999
  eval_copy(video_name, kind)
  loader = eval_loader()

  for index, x_real in enumerate(loader):
    num_frames = x_real.shape[2]
    x_real = x_real.to(device)
    x_fake = G.eval()(x_real)
    y_pred = D.eval()(x_real)
    x_real_frames = x_real.detach().to("cpu").unbind(dim=2)
    x_fake_frames = x_fake.detach().to("cpu").unbind(dim=2)
    pred_frames = y_pred.detach().to("cpu").unbind(dim=2)

    for i in range(num_frames):
      anomaly_count = (pred_frames[i] < threshold).sum()
      title = "{} : {} ".format(index*num_frames+i, anomaly_count)
      print(title)
      view_img([
                  tanh2sigmoid(x_real_frames[i][0]),
                  tanh2sigmoid(x_fake_frames[i][0]),  
                  pred_frames[i][0][0]
                  ], heat_index=[2])
      


