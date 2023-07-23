import os
import sys
import math
import time
import timm
import torch

import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

WORLD_SIZE = 4
BATCH_SIZES = {'s': 256, 'm': 96, 'l': 48}
MODEL_SIZE = sys.argv[1]
BATCH_SIZE = BATCH_SIZES[MODEL_SIZE]
EPOCHS = [10, 490]
SPLIT = sys.argv[2]
LOAD_CHECKPOINT = None if sys.argv[3] == 'None' else int(sys.argv[3])
PORT = sys.argv[4]
checkpoints_dir = Path(f'{MODEL_SIZE}_{SPLIT}')
checkpoints_dir.mkdir(exist_ok = True)

def train(rank):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = PORT
    dist.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=rank)
    torch.backends.cudnn.benchmark = True
    print('GPU', rank, 'initialised', flush=True)
    
    model = timm.create_model(f'tf_efficientnetv2_{MODEL_SIZE}_in21k', pretrained=(LOAD_CHECKPOINT is None), num_classes=12530).to(rank)
    torch.cuda.set_device(rank)
    model = torch.compile(nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=(LOAD_CHECKPOINT < sum(EPOCHS[:-1]) if LOAD_CHECKPOINT is not None else True)))
    criterion = nn.CrossEntropyLoss().to(rank)
    print('GPU', rank, 'model created', flush=True)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_dataset = datasets.ImageFolder(
        f'dataset/{SPLIT}',
        transforms.Compose([
            transforms.RandomResizedCrop(300 if MODEL_SIZE == 's' else 384),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24, sampler=train_sampler)
    print('GPU', rank, 'loader ready', flush=True)

    steps_per_epoch = len(train_loader)
    current_epoch = 0
    model.requires_grad_(False)

    for stage_i, epochs_i in enumerate(EPOCHS):

        if stage_i == 0:
            scaler = amp.GradScaler()
            print('GPU', rank, 'training started', flush=True)
            model.module.classifier.requires_grad_(True)
            if rank == 0:
                writer = SummaryWriter(str(checkpoints_dir / str(rank)))
                current_time = time.time()
        elif stage_i == len(EPOCHS) - 1:
            model.module.requires_grad_(True)
        else:
            if stage_i == 1:
                model.module.bn2.requires_grad_(True)
                model.module.conv_head.requires_grad_(True)
            model.module.blocks[-stage_i].requires_grad_(True)
        if LOAD_CHECKPOINT is not None and current_epoch + epochs_i <= LOAD_CHECKPOINT:
            current_epoch += epochs_i
            continue
        optimizer = RMSprop(params = filter(lambda p: p.requires_grad, model.parameters()), lr = 2.5e-6 * WORLD_SIZE * BATCH_SIZE / 16, alpha=0.9, eps=1e-08, weight_decay=1e-5, momentum=0.9)
        if LOAD_CHECKPOINT is not None and current_epoch <= LOAD_CHECKPOINT:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(f'{checkpoints_dir}/checkpoint_epoch{LOAD_CHECKPOINT}.pth', map_location = map_location)
            model.module.load_state_dict(checkpoint['model'])
            if current_epoch < LOAD_CHECKPOINT:
                optimizer.load_state_dict(checkpoint['optim'])
            del checkpoint
        current_epoch_static = current_epoch
        lr_manager = LambdaLR(optimizer, lambda step: (0.99 ** (float(max(current_epoch_static, LOAD_CHECKPOINT if LOAD_CHECKPOINT is not None else 0)) + float(step) / float(steps_per_epoch))))

        if stage_i == 0 and rank == 0 and LOAD_CHECKPOINT is None:
            cp_path = checkpoints_dir / ("checkpoint_epoch0.pth")
            torch.save({
                'epoch': 0,
                'model': model.module.state_dict(),
                'optim': optimizer.state_dict()
            }, str(cp_path))
            print(f"Saved checkpoint to {str(cp_path)}")
        dist.barrier()

        for epoch in range(current_epoch, current_epoch + epochs_i):

            if LOAD_CHECKPOINT is not None and epoch < LOAD_CHECKPOINT:
                current_epoch += 1
                continue
            model.train()
            train_sampler.set_epoch(epoch)
            for step, data in enumerate(train_loader):
                inputs = data[0].cuda(rank, non_blocking=True)
                targets = data[1].cuda(rank, non_blocking=True)
                optimizer.zero_grad()
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                lr_manager.step()
                scaler.update()
                if rank == 0:
                    print(f"Epoch {epoch} Step {step} GPU {rank} Loss {loss.item():6.4f} Latency {time.time()-current_time:4.3f} LR {optimizer.param_groups[0]['lr']:1.8f}", flush=True)
                    current_time = time.time()
                    if step % 100 == 99:
                        writer.add_scalar('train/loss', loss.item(), steps_per_epoch * epoch + step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], steps_per_epoch * epoch + step)

            if epoch % 5 == 4:
                if rank == 0:
                    cp_path = checkpoints_dir / ("checkpoint_epoch" + str(epoch + 1) + ".pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model': model.module.state_dict(),
                        'optim': optimizer.state_dict()
                    }, str(cp_path))
                    print(f"Saved checkpoint to {str(cp_path)}")
                dist.barrier()
            current_epoch = epoch + 1

if __name__ == '__main__':
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    mp.spawn(train, nprocs = WORLD_SIZE)
