import os
import cv2

model_code = 13
hr_dcn = False
offset_prop = False
split_ratio = 3
model_name = 'FVSR_x8_simple_v{}_hrdcn_{}_offsetprop_{}_fnet{}'.format(model_code, 'y' if hr_dcn else 'n',
                                                                                   'y' if offset_prop else 'n', 
                                                                                   '_{}outof4'.format(4-split_ratio) if model_code == 18 else '')
# print('Current model name: {}'.format(model_name))

dir_root = 'test_png'
gt_root = '{}/GroundTruth/'.format(dir_root)
model_names = os.listdir(dir_root)
for model_name in model_names:
    if model_name == 'GroundTruth' or model_name == 'eval_video' or model_name == 'results':
        continue
    print('Current model name: {}'.format(model_name))
    img_root = '{}/{}/'.format(dir_root, model_name)
    save_root = '{}/{}/'.format(dir_root, 'results')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #### 000 Past Foveated Region
    video_num = 0
    img_0_num = 66
    img_1_num = 75
    img_dir_path = os.path.join(gt_root, str(video_num))
    save_path = os.path.join(save_root, model_name, 'pastfv')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    begin_x_0 = 20
    begin_y_0 = 350
    begin_x_1 = 550
    begin_y_1 = 350
    begin_x_2 = 5
    begin_y_2 = 210
    img_w = 350
    img_h = 350
    img_w_0 = 80
    img_h_0 = 80
    hr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
    hr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
    hr_img_0 = cv2.imread(hr_img_0)
    hr_img_1 = cv2.imread(hr_img_1)
    GT_H, GT_W, _ = hr_img_0.shape
    cv2.rectangle(hr_img_0, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), (255,  51, 153), 3)
    cv2.rectangle(hr_img_1, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), ( 51, 255, 153), 3)
    cv2.rectangle(hr_img_1, (begin_x_1, begin_y_1), (begin_x_1+img_w, begin_y_1+img_h), ( 51, 153, 255), 3)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_0_num)), hr_img_0)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_1_num)), hr_img_1)

    dirs = os.listdir(os.path.join(img_root, str(video_num)))
    for dir in dirs:
        if dir == 'traj.png':
            continue
        img_dir_path = os.path.join(img_root, str(video_num), dir)
        sr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
        sr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
        sr_img_0 = cv2.imread(sr_img_0)
        sr_img_1 = cv2.imread(sr_img_1)
        sr_img_2 = sr_img_1.copy()
        if dir == 'results':
            sr_img_0 = sr_img_0[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
            sr_img_1 = sr_img_1[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
            sr_img_2 = sr_img_2[begin_y_1:begin_y_1+img_h, begin_x_1:begin_x_1+img_w, :]
            sr_img_3 = sr_img_1[begin_y_2:begin_y_2+img_h_0, begin_x_2:begin_x_2+img_w_0, :].copy()
            sr_img_4 = sr_img_1.copy()
            cv2.rectangle(sr_img_4, (begin_x_2, begin_y_2), (begin_x_2+img_w_0, begin_y_2+img_h_0), (255,  51, 153), 3)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_3.png'.format(img_1_num, dir)), sr_img_3)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_1_line.png'.format(img_1_num, dir)), sr_img_4)
        else:
            H, W, _ = sr_img_0.shape
            a = H / GT_H
            b = W / GT_W
            sr_img_0 = sr_img_0[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
            sr_img_1 = sr_img_1[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
            sr_img_2 = sr_img_2[int(begin_y_1*a):int((begin_y_1+img_h)*a), int(begin_x_1*b):int((begin_x_1+img_w)*b), :]
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_0_num, dir)), sr_img_0)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}_1.png'.format(img_1_num, dir)), sr_img_1)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}_2.png'.format(img_1_num, dir)), sr_img_2)

    #### 011 Whole Region
    video_num = 11
    img_0_num = 30
    img_1_num = 36
    img_dir_path = os.path.join(gt_root, str(video_num))
    save_path = os.path.join(save_root, model_name, 'whole')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    begin_x_0 = 200
    begin_y_0 = 100
    begin_x_1 = 500
    begin_y_1 = 100
    begin_x_2 = 360
    begin_y_2 = 100
    begin_x_3 = 60
    begin_y_3 = 90
    begin_x_4 = 60
    begin_y_4 = 40
    begin_x_5 = 200
    begin_y_5 = 60
    img_w = 750
    img_h = 450
    img_w_ = 360
    img_h_ = 216
    img_w__ = 100
    img_h__ = 120
    hr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
    hr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
    hr_img_0 = cv2.imread(hr_img_0)
    hr_img_1 = cv2.imread(hr_img_1)
    hr_img_2 = hr_img_0.copy()
    hr_img_3 = hr_img_1.copy()
    GT_H, GT_W, _ = hr_img_0.shape
    cv2.rectangle(hr_img_0, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), (255,  51, 153), 3)
    cv2.rectangle(hr_img_1, (begin_x_1, begin_y_1), (begin_x_1+img_w, begin_y_1+img_h), ( 51, 153, 255), 3)
    cv2.rectangle(hr_img_2, (begin_x_2, begin_y_2), (begin_x_2+img_w, begin_y_2+img_h), (255,  51, 153), 3)
    cv2.rectangle(hr_img_3, (begin_x_2, begin_y_2), (begin_x_2+img_w, begin_y_2+img_h), ( 51, 153, 255), 3)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_0_num)), hr_img_0)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_1_num)), hr_img_1)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line_2.png'.format(img_0_num)), hr_img_2)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line_2.png'.format(img_1_num)), hr_img_3)

    dirs = os.listdir(os.path.join(img_root, str(video_num)))
    for dir in dirs:
        if dir == 'traj.png':
            continue
        img_dir_path = os.path.join(img_root, str(video_num), dir)
        sr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
        sr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
        sr_img_0 = cv2.imread(sr_img_0)
        sr_img_1 = cv2.imread(sr_img_1)
        sr_img_2 = sr_img_0.copy()
        sr_img_3 = sr_img_1.copy()
        if dir == 'results':
            sr_img_0 = sr_img_0[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
            sr_img_1 = sr_img_1[begin_y_1:begin_y_1+img_h, begin_x_1:begin_x_1+img_w, :]
            sr_img_2 = sr_img_2[begin_y_2:begin_y_2+img_h, begin_x_2:begin_x_2+img_w, :]
            sr_img_3 = sr_img_3[begin_y_2:begin_y_2+img_h, begin_x_2:begin_x_2+img_w, :]
            cv2.rectangle(sr_img_2, (begin_x_3, begin_y_3), (begin_x_3+img_w_, begin_y_3+img_h_), ( 51, 153, 255), 3)
            cv2.rectangle(sr_img_3, (begin_x_3, begin_y_3), (begin_x_3+img_w_, begin_y_3+img_h_), ( 51, 153, 255), 3)
            sr_img_4 = sr_img_2[begin_y_3:begin_y_3+img_h_, begin_x_3:begin_x_3+img_w_, :]
            sr_img_5 = sr_img_3[begin_y_3:begin_y_3+img_h_, begin_x_3:begin_x_3+img_w_, :]
            sr_img_6 = sr_img_5[begin_y_4:begin_y_4+img_h__, begin_x_4:begin_x_4+img_w__, :].copy()
            sr_img_7 = sr_img_5[begin_y_5:begin_y_5+img_h__, begin_x_5:begin_x_5+img_w__, :].copy()
            cv2.rectangle(sr_img_5, (begin_x_4, begin_y_4), (begin_x_4+img_w__, begin_y_4+img_h__), ( 51, 153, 255), 3)
            cv2.rectangle(sr_img_5, (begin_x_5, begin_y_5), (begin_x_5+img_w__, begin_y_5+img_h__), ( 51, 153, 255), 3)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_3.png'.format(img_0_num, dir)), sr_img_4)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_3.png'.format(img_1_num, dir)), sr_img_5)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_4.png'.format(img_1_num, dir)), sr_img_6)
            cv2.imwrite(os.path.join(save_path, '{:03d}_{}_5.png'.format(img_1_num, dir)), sr_img_7)
        else:
            H, W, _ = sr_img_0.shape
            a = H / GT_H
            b = W / GT_W
            sr_img_0 = sr_img_0[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
            sr_img_1 = sr_img_1[int(begin_y_1*a):int((begin_y_1+img_h)*a), int(begin_x_1*b):int((begin_x_1+img_w)*b), :]
            sr_img_2 = sr_img_2[int(begin_y_2*a):int((begin_y_2+img_h)*a), int(begin_x_2*b):int((begin_x_2+img_w)*b), :]
            sr_img_3 = sr_img_3[int(begin_y_2*a):int((begin_y_2+img_h)*a), int(begin_x_2*b):int((begin_x_2+img_w)*b), :]
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_0_num, dir)), sr_img_0)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_1_num, dir)), sr_img_1)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}_2.png'.format(img_0_num, dir)), sr_img_2)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}_2.png'.format(img_1_num, dir)), sr_img_3)

    #### 015 Title
    video_num = 15
    img_0_num = 31
    img_1_num = 36
    img_2_num = 43
    img_dir_path = os.path.join(gt_root, str(video_num))
    save_path = os.path.join(save_root, model_name, 'title')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    begin_x_0 = 350
    begin_y_0 = 140
    begin_x_1 = 700
    begin_y_1 = 140
    img_w = 500
    img_h = 300
    hr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
    hr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
    hr_img_2 = os.path.join(img_dir_path, '{:03d}.png'.format(img_2_num))
    hr_img_0 = cv2.imread(hr_img_0)
    hr_img_1 = cv2.imread(hr_img_1)
    hr_img_2 = cv2.imread(hr_img_2)
    GT_H, GT_W, _ = hr_img_0.shape
    cv2.rectangle(hr_img_0, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), ( 255, 51, 153), 3)
    cv2.rectangle(hr_img_1, (begin_x_1, begin_y_1), (begin_x_1+img_w, begin_y_1+img_h), ( 51, 153, 255), 3)
    cv2.rectangle(hr_img_2, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), ( 51, 153, 255), 3)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_0_num)), hr_img_0)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_1_num)), hr_img_1)
    cv2.imwrite(os.path.join(save_path, '{:03d}_hr_line.png'.format(img_2_num)), hr_img_2)

    dirs = os.listdir(os.path.join(img_root, str(video_num)))
    for dir in dirs:
        if dir == 'traj.png':
            continue
        img_dir_path = os.path.join(img_root, str(video_num), dir)
        sr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
        sr_img_1 = os.path.join(img_dir_path, '{:03d}.png'.format(img_1_num))
        sr_img_2 = os.path.join(img_dir_path, '{:03d}.png'.format(img_2_num))
        sr_img_0 = cv2.imread(sr_img_0)
        sr_img_1 = cv2.imread(sr_img_1)
        sr_img_2 = cv2.imread(sr_img_2)
        if dir == 'results':
            sr_img_0 = sr_img_0[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
            sr_img_1 = sr_img_1[begin_y_1:begin_y_1+img_h, begin_x_1:begin_x_1+img_w, :]
            sr_img_2 = sr_img_2[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
        else:
            H, W, _ = sr_img_0.shape
            a = H / GT_H
            b = W / GT_W
            sr_img_0 = sr_img_0[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
            sr_img_1 = sr_img_1[int(begin_y_1*a):int((begin_y_1+img_h)*a), int(begin_x_1*b):int((begin_x_1+img_w)*b), :]
            sr_img_2 = sr_img_2[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_0_num, dir)), sr_img_0)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_1_num, dir)), sr_img_1)
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_2_num, dir)), sr_img_2)

    #### 020 Gaussian
    video_num = 20
    img_0_num = 99
    img_dir_path = os.path.join(img_root, str(video_num), 'results')
    save_path = os.path.join(save_root, model_name, 'title')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    begin_x_0 = 320
    begin_y_0 = 180
    img_w = 640
    img_h = 360
    hr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
    hr_img_0 = cv2.imread(hr_img_0)
    GT_H, GT_W, _ = hr_img_0.shape
    cv2.rectangle(hr_img_0, (begin_x_0, begin_y_0), (begin_x_0+img_w, begin_y_0+img_h), ( 255, 51, 153), 3)
    cv2.imwrite(os.path.join(save_path, '{:03d}_sr_line.png'.format(img_0_num)), hr_img_0)

    dirs = os.listdir(os.path.join(img_root, str(video_num)))
    for dir in dirs:
        if dir == 'traj.png':
            continue
        img_dir_path = os.path.join(img_root, str(video_num), dir)
        sr_img_0 = os.path.join(img_dir_path, '{:03d}.png'.format(img_0_num))
        sr_img_0 = cv2.imread(sr_img_0)
        if dir == 'results':
            sr_img_0 = sr_img_0[begin_y_0:begin_y_0+img_h, begin_x_0:begin_x_0+img_w, :]
        else:
            H, W, _ = sr_img_0.shape
            a = H / GT_H
            b = W / GT_W
            sr_img_0 = sr_img_0[int(begin_y_0*a):int((begin_y_0+img_h)*a), int(begin_x_0*b):int((begin_x_0+img_w)*b), :]
        cv2.imwrite(os.path.join(save_path, '{:03d}_{}.png'.format(img_0_num, dir)), sr_img_0)