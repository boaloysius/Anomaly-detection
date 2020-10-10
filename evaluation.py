from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from utils import *
import pandas as pd

from evaluate_model import eval_copy, eval_loader
from evaluation import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ssim_skimage(pred_frame, x_real_frame, win_size=99):
  ssim = compare_ssim(
              tanh2sigmoid(pred_frame.permute(1,2,0)).numpy(),  
              tanh2sigmoid(x_real_frame.permute(1,2,0)).numpy(),
              win_size=win_size, 
              multichannel=True
          )
  return 1-ssim

def get_psnr_skimage(pred_frame, x_real_frame):
  psnr = compare_psnr(
              tanh2sigmoid(pred_frame.permute(1,2,0)).numpy(),  
              tanh2sigmoid(x_real_frame.permute(1,2,0)).numpy(),
              )
  return 1-psnr

def image_denormalise(x):
  return ((x.permute(1,2,0)*0.5+0.5)*255)

def tensor2print(x):
  return x.numpy().astype(np.uint8)

score_map = {"ssim_score":get_ssim_skimage,
             "psnr_score":get_psnr_skimage
             }

def get_eval_loader(dataset_name="UCSDped1"):
  #print(dataset_name)
  if(dataset_name=="UCSDped1"):
    eval_folders=["Test003", "Test014", "Test018", "Test019", "Test021", "Test022", "Test023", "Test024", "Test032"]
  if(dataset_name=="UCSDped2"):
    eval_folders=["Test001", "Test002", "Test003", "Test004", "Test005", "Test006", "Test007", "Test008", "Test009", "Test010", "Test011", "Test012"]
  
  selected = eval_folders
  eval_copy(selected, dataset_name)
  loader = eval_loader(dataset_name)
  return loader

def get_eval_df(loader, G):

  processed_eval_list = []
  G.to(device)
  with torch.no_grad():
    for index, data in enumerate(loader):
      if(index%100==0):
        print(index)
      x_real, target, gt, file_names = data

      #print(file_names)
      num_frames = x_real.shape[2]
      pred_G = G(x_real.to(device))
      pred_frames_G = pred_G.detach().to("cpu").unbind(dim=2)
      x_real_frames = x_real.detach().to("cpu").unbind(dim=2)
      
      for i in range(num_frames):
        
        frame_data = {}
        for key in score_map:
          fn = score_map[key]
          frame_data[key] = fn(pred_frames_G[i][0], x_real_frames[i][0]) 
        frame_data["target"] = target[i].numpy()[0] 
        frame_data["video_name"] = file_names[i][0].split("/")[0]
        frame_data["file_name"] = file_names[i]

        # Add frames to video
        frame_data["x_real"] = tensor2print(image_denormalise(x_real_frames[i][0]))
        frame_data["x_mask"] = tensor2print(gt[i][0])
        frame_data["x_pred"] = tensor2print(image_denormalise(pred_frames_G[i][0]))

        processed_eval_list.append(frame_data)


  eval_df = pd.DataFrame(processed_eval_list)
  for score_key in score_map:
    eval_df[score_key+'_nor'] = eval_df.groupby('video_name')[score_key].apply(lambda x: (x-x.min())/(x.max()-x.min()))

  return eval_df

def evaluate_model(G, name="eval", drive=False, dataset_name="UCSDped1"):
  loader = get_eval_loader(dataset_name)
  eval_df = get_eval_df(loader, G)
  from sklearn.metrics import roc_auc_score
  auc_scores = {}
  for key in score_map:
    auc_scores[key]=np.round(roc_auc_score(eval_df["target"], eval_df[key+"_nor"]), 2)

  eval_video_frames = []
  for _,row in eval_df.iterrows():
    eval_video_frames.append(combine_eval_frame(row, "AUC:{},{}".format(auc_scores["ssim_score"], auc_scores["psnr_score"])))

  write_video(eval_video_frames,"{}_{}.mp4".format(name, auc_scores["ssim_score"]), drive)
  return auc_scores