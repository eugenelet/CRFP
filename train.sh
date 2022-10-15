#### Train BasicVSR with Reds
python3 main.py --save_dir ./train/REDS/FVSR_x8_simple_v1_dcn2_v11 \
               --reset True \
               --log_file_name train.log \
               --num_gpu 4 \
               --gpu_id 0 \
               --num_workers 9 \
               --dataset Reds \
               --dataset_dir /DATA/REDS_sharp/ \
               --model_path ./train/REDS/FVSR_x8_simple_v1_dcn2_v10/model/model_00002_005600.pt \
               --n_feats 64 \
               --lr_rate 2e-4 \
               --lr_rate_flow 2.5e-5 \
               --rec_w 1 \
               --scale 8 \
               --cra true \
               --mrcf true \
               --batch_size 8 \
               --FV_size 128 \
               --GT_size 256 \
               --N_frames 15 \
               --y_only false \
               --num_init_epochs 2 \
               --num_epochs 80 \
               --print_every 200 \
               --save_every 100 \
               --val_every 1 \
               --visdom_port 8803 \
               --visdom_view 1227_FVSR_x8_simple_v1_dcn2_v11

### simple_v1  dk=1, fd=32-> 8, lv1, range=10, dcn_gp=16
### simple_v2  dk=3, fd=32-> 8, lv1, range=10, dcn_gp=16
### simple_v3  dk=3, fd=64->16, lv1, range=10, dcn_gp=16

### simple_v4  dk=1, fd=32-> 8, lv3, range=10, dcn_gp= 1
### simple_v5  dk=3, fd=32-> 8, lv3, range=10, dcn_gp= 1
### simple_v6  dk=3, fd=32-> 8, lv3, range=80, dcn_gp= 1
### simple_v7  dk=3, fd=32-> 8, lv3, range=80, dcn_gp= 4

### simple_v8  dk=3, fd=64->16, lv3, range=80, dcn_gp= 4
### simple_v9  dk=3, fd=64-> 8, lv3, range=80, dcn_gp= 4
### simple_v10 dk=3, fd=32->16, lv3, range=80, dcn_gp= 4
### simple_v11 dk=1, fd=32->16, lv3, range=80, dcn_gp=16

### simple_duf dk=1, fd=32-> 8, lv1, range=10, dcn_gp=16

### simple_v12 dk=1, fd=32->32, lv3, range=80, dcn_gp= 4
### simple_v13 dk=1, fd=32->32, lv3, range=80, dcn_gp= 4, dcn * 2, res * 2

### dcn3_v1 dcn * 3, dk=1
### dcn3_v2 dcn * 3, dk=3(offset using repeat)
### dcn3_v3 dcn * 3, dk=3(offset & mask using repeat)

### dcn2_v1  dcn * 2, dk=1
### dcn2_v2  dcn * 2, dk=1, offset finetune(mean of generated offset)
### dcn2_v3  dcn * 2, dk=1, upsample * 2
### dcn2_v4  dcn * 2, dk=1, res_block * 2
### dcn2_v5  dcn * 2, dk=1, branch out
### dcn2_v6  dcn * 2, dk=1, deeper downsampel layer(conv2d * 2)
### dcn2_v7  dcn * 2, dk=1, deeper downsampel layer(conv2d * 4)
### dcn2_v8  dcn * 2, dk=1, v4 deeper downsampel layer(PS * 2)
### dcn2_v9  dcn * 2, dk=1, channel dimension = 32
### dcn2_v10 dcn * 4, dk=1, channel dimension = 32
### dcn2_v11 pca    , dk=1, channel dimension = 32