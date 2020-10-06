import torch
from torchvision.transforms import transforms
import torchvision
from PIL import Image
import sys
import numpy as np
import glob
import bisect

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, resize_size=None, resize_mode=Image.BILINEAR, rgb=False, depth=5):
        super(torch.utils.data.Dataset, self).__init__()
        self.path = path
        
        videos = sorted(glob.glob(path+"/*"))
        self.video_frameList = {video:sorted(glob.glob(video+"/*.png")) for video in videos}
        self.video_setCount = {video: int(len(frames)/depth) for video, frames in self.video_frameList.items() if int(len(frames)/depth) > 0} 
        self.videos = list(self.video_setCount.keys())
        setCount = list(self.video_setCount.values())
        self.cum_setCount = np.cumsum(setCount)
        self.min_indices = np.concatenate(([0],self.cum_setCount[:-1]))

        self.transform = self._make_transforms(resize_size, resize_mode)
        self.rgb = rgb
        self.depth = depth
        self.total_setCount = sum(self.video_setCount.values())

    def convert_index(self,index):
        i=bisect.bisect(self.cum_setCount,index)
        video = self.videos[i]
        base_index = self.min_indices[i]
        return(video, base_index)
        
    def __getitem__(self, _index):
        video, base_index = self.convert_index(_index)
        whole_video = self.video_frameList[video]
        index = (_index-base_index)*self.depth
        #print(_index, index, video)
        img_list = []
        anomaly_indicators = []
        gt_frames = []
        for i in range(index, index+self.depth):
          path = whole_video[i]
          #print(path)

          img = Image.open(path)
          if self.rgb:
              img = img.convert('RGB')
          else:
              img = img.convert('L')
          img_list.append(img)

          # Loading Target
          tmp_gt_path = whole_video[i]
          tmp_gt_path_split = tmp_gt_path.split("/")
          tmp_gt_path_split[-2] = tmp_gt_path_split[-2]+"_gt"
          gt_path =  ("/".join(tmp_gt_path_split)).replace(".png",".bmp")
          gt_img = np.array(Image.open(gt_path))
          gt_frames.append(gt_img)
          anomaly_indicators.append(int(gt_img.max()>0))
          print(path,"\n", gt_path)

        X = self.transform(img_list)
        
        
        return X,1
        
    def __len__(self):
        return self.total_setCount
    
    @staticmethod
    def _make_transforms(resize_size, resize_mode):
        transform = []
        if resize_size is not None:
            transform.append(GroupScale(resize_size, resize_mode))
        transform.append(Stack())
        transform.append(ToTorchFormatTensor())
        transform.append(GroupNormalize())
        transform = transforms.Compose(transform)
        return transform

class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupNormalize(object):
    def __init__(self):
        self.worker = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    def __call__(self, xs):
        xs = torch.unbind(xs, dim=1)
        xs = torch.stack([self.worker(x) for x in xs], dim=1)
        return xs


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
        # img: [C, L, H, W]
        return img.float().div(255) if self.div else img.float()

class Stack(object):
    def __init__(self):
      pass
    def __call__(self, pic):
      return np.stack(pic)
