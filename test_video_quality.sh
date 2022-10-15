#!/bin/bash

# file_dir="FVSR_x8_simple_v0_hrdcn_n_offsetprop_n_fnet"
# file_dir="FVSR_x8_simple_v13_hrdcn_n_offsetprop_n_fnet"
# file_dir="FVSR_x8_simple_v13_hrdcn_n_offsetprop_n_fnet_nodcn"
# file_dir="FVSR_x8_simple_v15_hrdcn_n_offsetprop_n_fnet"
# file_dir="FVSR_x8_simple_v15_hrdcn_n_offsetprop_y_fnet"
# file_dir="FVSR_x8_simple_v15_hrdcn_y_offsetprop_n_fnet"
# file_dir="FVSR_x8_simple_v15_hrdcn_y_offsetprop_y_fnet"
# file_dir="FVSR_x8_simple_v18_hrdcn_y_offsetprop_y_fnet_1outof4"
# file_dir="FVSR_x8_simple_v18_hrdcn_y_offsetprop_y_fnet_1outof4_nofv"
# file_dir="FVSR_x8_simple_v18_hrdcn_y_offsetprop_y_fnet_1outof4_fast"
file_dir="FVSR_x8_simple_v15_hrdcn_y_offsetprop_y_fnet_gaussian"
# file_dir="FVSR_x8_simple_v18_hrdcn_y_offsetprop_y_fnet_1outof4_gaussian"
# file_dir="FVSR_x8_simple_v18_hrdcn_y_offsetprop_y_fnet_1outof4_gaussian_regional"
# file_dir="Bicubic"
video_num=$1
ffmpeg \
    -r 24 -i test_video/$file_dir/$video_num/gt.mp4 \
    -r 24 -i test_video/$file_dir/$video_num/sr.mp4 \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]scale=1280:720:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout:model_path=/home/si2/vmaf/model/vmaf_v0.6.1.json" \
    -f null - > test_video/$file_dir/$video_num/eval.log
echo $file_dir