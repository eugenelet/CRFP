#### take reference from 

from option import args
from utils import mkExpDir
from dataset import dataloader
from model import CRFP
from loss.loss import get_loss_dict
from trainer import Trainer

import math
import os
import time
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    ### make save_dir
    _logger = mkExpDir(args)

    ### device and model
    if args.num_gpu == 1:
        device = torch.device('cpu') if args.cpu else torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu') if args.cpu else torch.device('cuda')

    # _model = CRFP.BasicFVSR(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # _model = CRFP.CRFP_simple_noDCN(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # _model = CRFP.CRFP_simple(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # _model = CRFP.CRFP(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    _model = CRFP.CRFP_DSV(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # _model = CRFP.CRFP_DSV_CRA(mid_channels=32, y_only=args.y_only, hr_dcn=args.hr_dcn, offset_prop=args.offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)

    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args)

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)
    t.before_run()
    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test_basicvsr()
    elif (args.eval):
        model_list = sorted(os.listdir(args.model_path))
        # model_list = model_list[::-1]
        for idx, m in enumerate(model_list):
            t.load(model_path=os.path.join(args.model_path, m))
            t.eval_basicvsr(idx)
            t.vis_plot_metric('eval')
    else:
        # t.load(model_path=args.model_path)
        for epoch in range(1, args.num_epochs+1):
            t.train_basicvsr(current_epoch=epoch)
            t.vis_plot_metric('train')
            # if (epoch % args.val_every == 0):
                # t.eval_basicvsr(current_epoch=epoch)
                # t.vis_plot_metric('eval')
        torch.cuda.empty_cache()