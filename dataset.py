import torch
from torchvision.transforms import transforms
import torchvision
from PIL import Image
import sys
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, resize_size=None, resize_mode=Image.BILINEAR, rgb=False):
        super(torch.utils.data.Dataset, self).__init__()
        self.paths = paths
        self.transform = self._make_transforms(resize_size, resize_mode)
        self.rgb = rgb
        
    def __getitem__(self, index):
        img_list = []
        for i in range(index, index+5):
          path = self.paths[i]
          img = Image.open(path)
          if self.rgb:
              img = img.convert('RGB')
          else:
              img = img.convert('L')
          img_list.append(img)

        X = self.transform(img_list)
        return X
        
    def __len__(self):
        return len(self.paths) - 5
    
    @staticmethod
    def _make_transforms(resize_size, resize_mode):
        transform = []
        if resize_size is not None:
            transform.append(GroupScale(resize_size, resize_mode))
        transform.append(Stack())
        transform.append(ToTorchFormatTensor())
        #transform.append(transforms.ToTensor())
        #transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform = transforms.Compose(transform)
        return transform

class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        # handle PIL Image
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.mode))
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")
