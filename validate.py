import os
import sys
import math
import timm
import torch
import pickle

import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

def confidence_mask(probabilities, threshold):
    return probabilities.max(1)[0] > threshold

def margin_mask(probabilities, threshold):
    top_probabilities, _ = torch.topk(probabilities, 2, 1)
    return top_probabilities[:, 0] - top_probabilities[:, 1] > threshold

def entropy_mask(probabilities, threshold):
    return 1 + (probabilities * torch.log(probabilities).nan_to_num()).sum(1) / math.log(probabilities.shape[1]) > threshold

splits = ['train', 'test']
abstentions = {'confidence': confidence_mask, 'margin':margin_mask, 'entropy':entropy_mask}
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
model_size = sys.argv[1]
checkpoints_dir = sys.argv[2]
model = timm.create_model(f'tf_efficientnetv2_{model_size}', pretrained=False, num_classes=12530).cuda()
criterion = nn.CrossEntropyLoss().cuda()
softmax = nn.Softmax(dim=-1).cuda()

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tools = {}
for split in splits:
    tools[split] = {}
    tools[split]['loader'] = torch.utils.data.DataLoader(datasets.ImageFolder(
        f"dataset_{384 if model_size == 's' else 480}/{split}",
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    ), batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    tools[split]['writer_summary']= SummaryWriter(f'{checkpoints_dir}/test/summary_{split}')
    for abstention in abstentions:
        tools[split][f'writer_{abstention}'] = SummaryWriter(f'{checkpoints_dir}/test/{abstention}_{split}')
assert tools['train']['loader'].dataset.class_to_idx == tools['test']['loader'].dataset.class_to_idx
instance_count = pickle.load(open('instance_count.pkl', 'rb'))
class_bins = {v: (0 if instance_count[k] < 5 else (1 if instance_count[k] < 10 else (2 if instance_count[k] < 20 else (3 if instance_count[k] < 50 else 4)))) for k, v in tools['train']['loader'].dataset.class_to_idx.items()}
for epoch in range(int(sys.argv[3]), int(sys.argv[4]) + 5, 5):
    checkpoint = torch.load(f'{checkpoints_dir}/checkpoint_epoch{epoch}.pth')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.eval()
    for split in splits:
        with open(f'{checkpoints_dir}/test/{split}_{epoch}.csv', 'w') as fp:
            fp.write('true_label,pred_label,confidence\n')
            running_loss = 0.0
            processed = 0
            top_correct = [0 for _ in range(5)]
            processed_binned = [0 for _ in range(5)]
            correct_binned = [0 for _ in range(5)]
            abstention_processed = {abstention: [0 for _ in thresholds] for abstention in abstentions}
            abstention_correct = {abstention: [0 for _ in thresholds] for abstention in abstentions}
            with torch.no_grad():
                for step, data in enumerate(tools[split]['loader']):
                    inputs = data[0].cuda()
                    targets = data[1].cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()
                    processed += targets.size(0)
                    probabilities = softmax(outputs)
                    confidence, predicted = torch.topk(probabilities, 5, 1)
                    for tl_i, pl_i, c_i in zip(targets.cpu(), predicted[:, 0].cpu(), confidence[:, 0].cpu()):
                        fp.write(f'{int(tl_i)},{int(pl_i)},{float(c_i)}\n')
                    correct_labels = predicted == targets.unsqueeze(1)
                    for abstention in abstentions:
                        for i, threshold in enumerate(thresholds):
                            mask = abstentions[abstention](probabilities, threshold)
                            abstention_processed[abstention][i] += mask.sum().item()
                            abstention_correct[abstention][i] += (correct_labels[:, 0] * mask).sum().item()
                    top_correct = [correct_labels[:, :k + 1].sum().item() + tc for k, tc in enumerate(top_correct)]
                    targets_binned = [class_bins[target] for target in targets.cpu().tolist()]
                    correct_list = correct_labels[:, 0].cpu().tolist()
                    for tb, cl in zip(targets_binned, correct_list):
                        processed_binned[tb] += 1
                        correct_binned[tb] += cl
                print(epoch, split, [float(f"{100.0*tc/processed:3.3f}") for tc in top_correct], flush=True)
            tools[split]['writer_summary'].add_scalar('summary/loss', running_loss / (step + 1), epoch)
            for k, bin_range in enumerate(['1_4', '5_9', '10_19', '20_49', '50_']):
                tools[split]['writer_summary'].add_scalar(f'summary/{bin_range}', 100.0 * correct_binned[k] / processed_binned[k], epoch)
            for k in range(5):
                tools[split]['writer_summary'].add_scalar(f'summary/top{k+1:d}', 100.0 * top_correct[k] / processed, epoch)
            for abstention in abstentions:
                for i, threshold in enumerate(thresholds):
                    tools[split][f'writer_{abstention}'].add_scalar(f'abstention/acc_wrt_predicted_epoch_{epoch:d}', 0.0 if abstention_processed[abstention][i] == 0 else 100.0 * abstention_correct[abstention][i] / abstention_processed[abstention][i], round(100 * abstention_processed[abstention][i] / processed))
