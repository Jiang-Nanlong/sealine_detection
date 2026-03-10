# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py

Entropy support:
  当 dataloader batch 含 entropy_maps 时（3-tuple 或 4-tuple），
  engine 自动将 entropy_maps 传给 model(samples, targets, entropy_map=...)。
  当 dataloader 返回 2-tuple (samples, targets) 时，行为与原始完全一致。
"""

import math
import sys
from typing import Iterable

import torch
import util.misc as utils


def _unpack_batch(batch):
    """从 dataloader batch 中解包，兼容 2 / 3 / 4-tuple。
    始终返回 (samples, targets, entropy_maps_or_None)。
    第 4 个元素 metas（如有）在此处丢弃，不参与训练。
    """
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None,
                    lr_scheduler=None, warmup_scheduler=None, args=None, ema_m=None):
    scaler = torch.amp.GradScaler(str(device), enabled=args.amp)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets, entropy_maps = _unpack_batch(batch)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if entropy_maps is not None:
            entropy_maps = entropy_maps.to(device)

        global_step = epoch * len(data_loader) + i

        with torch.amp.autocast(str(device), enabled=args.amp):
            if entropy_maps is not None:
                outputs = model(samples, targets, entropy_map=entropy_maps)
            else:
                outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch in metric_logger.log_every(data_loader, 250, header):
        samples, targets, entropy_maps = _unpack_batch(batch)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if entropy_maps is not None:
            entropy_maps = entropy_maps.to(device)

        with torch.amp.autocast(str(device), enabled=args.amp):
            if entropy_maps is not None:
                outputs = model(samples, targets, entropy_map=entropy_maps)
            else:
                outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        metric_logger.update(loss=sum(loss_dict_reduced.values()),
                             **loss_dict_reduced,)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats


@torch.no_grad()
def test(model, criterion, postprocessors, evaluator, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    evaluator.cleanup()

    header = 'Test:'

    for batch in metric_logger.log_every(data_loader, 250, header):
        samples, targets, entropy_maps = _unpack_batch(batch)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if entropy_maps is not None:
            entropy_maps = entropy_maps.to(device)

        if entropy_maps is not None:
            outputs = model(samples, targets, entropy_map=entropy_maps)
        else:
            outputs = model(samples, targets)

        evaluator.update(outputs, targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.accumulate()
    evaluator.summarize()

    return