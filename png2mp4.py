
import os
import numpy as np
from PIL import Image
import cv2
from torchaudio import save_encinfo

if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    model_code = 15
    hr_dcn = True
    offset_prop = True
    split_ratio = 3
    model_name = 'FVSR_x8_simple_v{}_hrdcn_{}_offsetprop_{}_fnet{}_gaussian'.format(model_code, 'y' if hr_dcn else 'n',
                                                                                       'y' if offset_prop else 'n', 
                                                                                       '_{}outof4'.format(4-split_ratio) if model_code == 18 else '')
    # model_name = 'Bicubic'

    video_nums = [0, 11, 15, 20]
    for video_num in video_nums: 
        sr_png_dir = 'test_png/eval_video/{}/{}/results'.format(model_name, video_num)
        gt_png_dir = 'test_png/eval_video/GroundTruth/{}/'.format(video_num)

        save_dir = 'test_video/{}/{}'.format(model_name, video_num)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        gt_imgs = []
        sr_imgs = []
        files = os.listdir(sr_png_dir)
        files = sorted(files)
        for f in files:
            if '.gif' in f:
                continue
            img = cv2.imread(os.path.join(sr_png_dir, f))
            sr_imgs.append(img)
        files = os.listdir(gt_png_dir)
        files = sorted(files)
        for f in files:
            img = cv2.imread(os.path.join(gt_png_dir, f))
            gt_imgs.append(img)

        H, W, C = sr_imgs[0].shape
        out = cv2.VideoWriter(os.path.join(save_dir, 'sr.mp4'), fourcc, 20.0, (W,  H))
        for i in range(len(sr_imgs)):
            out.write(sr_imgs[i])

        H, W, C = gt_imgs[0].shape
        out = cv2.VideoWriter(os.path.join(save_dir, 'gt.mp4'), fourcc, 20.0, (W,  H))
        for i in range(len(gt_imgs)):
            out.write(gt_imgs[i])