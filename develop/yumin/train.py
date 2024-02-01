import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import numpy as np
np.random.seed(2024)

import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR','save_pth/_extended_experiment'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('--ignore_list', type=list, default=['drp.en_ko.in_house.deepnatural_002491.jpg',
                                                            'drp.en_ko.in_house.deepnatural_003347.jpg'])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, ignore_list):
    
    wandb.init(
    project="yumin", group = "level2-cv-10-detection", name='_extended_experiment',  # 변경 !!
    config={
        "learning_rate": args.learning_rate,  # 학습률을 wandb config에 추가
        "epochs": args.max_epoch,
    },
    )

    # 훈련 데이터셋 load
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        color_jitter = True,
        normalize = True,
        ignore_list = ignore_list
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 검증 데이터셋 로드
    # valid_dataset = SceneTextDataset(
    #     data_dir,
    #     split='valid',  # 'valid' 스플릿 사용
    #     image_size=image_size,
    #     crop_size=input_size,
    #     ignore_tags=ignore_tags
    # )
    # valid_dataset = EASTDataset(valid_dataset)
    # _num_batches = math.ceil(len(valid_dataset) / batch_size)
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers
    # )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False, saved_model_path='./save_pth/only_t_extended_experiment/50.pth')
    # model = EAST()
    model.to(device)
    print('cuda 사용중 :', torch.cuda.is_available())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 3, max_epoch//3*2], gamma=0.01)
    cnt = 1
    #model.train()
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start, _epoch_loss = 0, time.time(), 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    't_Cls loss': extra_info['cls_loss'], 't_Angle loss': extra_info['angle_loss'],
                    't_IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)


        # model.eval()
        # with tqdm(total=_num_batches) as _pbar:
        #     with torch.no_grad():
        #         for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
        #             _, _extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

        #             _pbar.update(1)
        #             _valid_dict = {
        #                 'v_Cls loss': _extra_info['cls_loss'], 'v_Angle loss': _extra_info['angle_loss'],
        #                 'v_IoU loss': _extra_info['iou_loss']
        #             }
        #             _pbar.set_postfix(_valid_dict)

        wandb.log({'learning_rate': optimizer.param_groups[0]['lr'],
                   't_Cls loss': extra_info['cls_loss'],
                   't_Angle loss': extra_info['angle_loss'],
                    't_IoU loss': extra_info['iou_loss']}#,
                    # "v_Cls loss": _extra_info['cls_loss'],
                    # "v_Angle loss": _extra_info['angle_loss'],
                    # 'v_IoU loss': _extra_info['iou_loss']}
                    )
        
        scheduler.step()
        cnt += 1

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            print('Vlid loss: {:.4f} | Elapsed time: {}'.format(
                _epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    wandb.finish()

def main(args):
    do_training(**args.__dict__)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
