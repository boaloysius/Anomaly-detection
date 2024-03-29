import os
import glob
import numpy as np
from PIL import Image
import sys

def preprocess_video(dirs):
    for _dir in dirs:
        if os.path.isfile(_dir):
            continue
        if _dir[-3:] == '_gt':
            continue           
        
        new_dir = _dir.replace('UCSD_Anomaly_Dataset.v1p2', 'UCSD_processed')
        os.makedirs(new_dir, exist_ok=True)
        dir_files = sorted(glob.glob(_dir + '/*.tif'))
        print('{} >>> {}'.format(_dir, new_dir))
        for file_ in dir_files:
            I_t = np.array(Image.open(file_), dtype='float')
            I = np.stack([I_t, I_t, I_t], axis=0)
            img = Image.fromarray(I.transpose(1, 2, 0).astype(np.uint8))
            save_path = file_.replace('UCSD_Anomaly_Dataset.v1p2', 'UCSD_processed').replace(".tif",".png")
            img.save(save_path)
            
            
if __name__ == '__main__':
    ped1_train_dirs = sorted(glob.glob('../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*'))
    ped2_train_dirs = sorted(glob.glob('../data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/*'))
    ped1_test_dirs = sorted(glob.glob('../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/*'))
    ped2_test_dirs = sorted(glob.glob('../data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/*'))

    preprocess_video(ped1_train_dirs)
    preprocess_video(ped2_train_dirs)
    preprocess_video(ped1_test_dirs)
    preprocess_video(ped2_test_dirs)