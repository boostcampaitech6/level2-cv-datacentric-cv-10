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

import wandb
# 필요한 라이브러리와 모듈을 가져옵니다.
# torch 및 관련 모듈은 딥러닝을 위한 것입니다.
# tqdm는 진행률 표시줄을 표시하는 데 사용됩니다.
# EASTDataset, SceneTextDataset 및 EAST는 사용자 정의 모듈입니다.

def parse_args():
    parser = ArgumentParser()

    # 이 함수는 커맨드 라인 실행을 위한 인자 파싱을 설정합니다.
    # 데이터 디렉토리, 모델 디렉토리 등 다양한 매개변수를 사용자 정의할 수 있습니다.

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
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    # 파서에 기본값과 설명이 있는 인자들을 추가합니다.
    # 이 인자들은 커맨드 라인을 통해 재정의될 수 있습니다.

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    
    # input_size가 EAST 모델 요구사항인 32의 배수인지 확인하기 위한 유효성 검사입니다.
    
    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags):
    
    # 이 함수는 핵심 훈련 함수입니다.
    # 디렉토리, 장치(CPU/GPU), 배치 크기, 학습률 등과 같은 여러 매개변수를 사용합니다.

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

    # 훈련을 위한 데이터셋을 로드하고 EASTDataset으로 래핑합니다. EASTDataset는 사용자 정의 DataLoader일 가능성이 높습니다.
    # DataLoader는 훈련을 위해 배치 단위로 데이터를 로드하는 PyTorch 유틸리티입니다.

    # W&B 초기화
    wandb.init(project="Project03_OCR", name="test_02", config={ # project명, test name 변경하고 사용
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "architecture": "EAST",
        "dataset": "EASTDataset",     
    })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # 장치 설정 (CPU 또는 GPU), EAST 모델 초기화 및 장치에 설정.
    # 훈련을 위한 옵티마이저와 학습률 스케줄러 정의.

    model.train()
    
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
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
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

                # W&B에 로그 기록
                wandb.log({"loss": loss_val, "epoch": epoch})

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        # 여기서 훈련 루프가 시작됩니다. 각 에포크마다 데이터셋을 배치 단위로 처리합니다.
        # 각 배치에 대해 순방향 패스를 수행하고, 손실을 계산하고, 모델 매개변수를 업데이트합니다.

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        # 지정된 간격으로 모델을 저장합니다.
    
    # W&B 실행 종료
    wandb.finish()

def main(args):
    do_training(**args.__dict__)

# 훈련 프로세스를 시작하는 메인 함수입니다.
# 인자를 풀고 훈련 함수를 호출합니다.
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

# 이것은 스크립트가 독립적인 프로그램으로 실행될 수 있도록 하는 파이썬 스크립트를 위한 표준 보일러플레이트입니다.

