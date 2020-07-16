import shutil
import os

from utils import *
import continuous_dataset

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
  dataset = continuous_dataset.Dataset(eval_dir, (size, size), rgb=True, depth=depth)
  loader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, num_workers=1)
  return loader

def evaluate_temporal_discriminator(model, video_name, kind="Test", threshold=None):
  if(threshold==None):
    threshold=0.999
  eval_copy(video_name, kind)
  loader = eval_loader()

  for index, x_real in enumerate(loader):
    frame_index = int(x_real.shape[2]/2)
    pred = model(x_real.to(device))[0][0][frame_index]
    anomaly_count = (pred < threshold).sum()
    title = "{} : {} ".format(index, anomaly_count)
    print(title)
    view_img([torch.unbind(x_real, dim=2)[frame_index][0], pred.detach().to("cpu").numpy() < threshold])