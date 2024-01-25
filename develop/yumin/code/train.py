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


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../../../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags):
    # best_loss = float('inf')

    # 훈련 데이터셋 load
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
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
    valid_dataset = SceneTextDataset(
        data_dir,
        split='valid',  # 'valid' 스플릿 사용
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    valid_dataset = EASTDataset(valid_dataset)
    _num_batches = math.ceil(len(valid_dataset) / batch_size)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    print('cuda 사용중 :', torch.cuda.is_available())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

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


        model.eval()
        valid_loss = 0
        with tqdm(total=_num_batches) as _pbar:
            with torch.no_grad():
                for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                    _loss, _extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    _loss_val = _loss.item()
                    _epoch_loss += _loss_val

                    _pbar.update(1)
                    _valid_dict = {
                        'v_Cls loss': _extra_info['cls_loss'], 'v_Angle loss': _extra_info['angle_loss'],
                        'v_IoU loss': _extra_info['iou_loss']
                    }
                    _pbar.set_postfix(_valid_dict)

        # 훈련 및 검증 메트릭을 모두 포함하여 출력
        #print(f'Epoch {epoch + 1} - Train Loss: {_epoch_loss / num_batches:.4f}, 'f'Valid Loss: {valid_loss:.4f}, {_valid_dict}')

        #print(f'Epoch {epoch+1}, Valid Loss: {valid_loss:.4f}')

        # 체크포인트 저장 조건
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     ckpt_fpath = osp.join(model_dir, 'best_model.pth')
        #     torch.save(model.state_dict(), ckpt_fpath)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            print('Vlid loss: {:.4f} | Elapsed time: {}'.format(
                _epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
