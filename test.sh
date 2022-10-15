### test

### test_video_with_reds
python3 main.py --save_dir ./test/demo/FVSR_x4_mrcf/output/ \
               --model_path ./train/model_00001_002900.pt \
               --log_file_name test.log \
               --reset True \
               --test True \
               --num_workers 1 \
               --scale 4 \
               --cra true \
               --mrcf true \
               --N_frames 15 \
               --FV_size 96 \
               --dataset Reds \
               --dataset_dir /DATA/REDS_sharp/ \
               --visdom_port 8803 \
               --visdom_view FVSR_x4_mrcf
