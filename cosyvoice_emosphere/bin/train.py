# 

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
try:
    import deepspeed
except Exception:
    deepspeed = None

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice_emosphere.utils.executor import Executor
from cosyvoice_emosphere.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        help='Start epoch index (useful when continuing training with torch_ddp checkpoints that only store weights).')
    parser.add_argument('--resume_train_step',
                        default=None,
                        type=int,
                        help='Global train step counter after which to resume (matches epoch_*_whole.yaml "step:" at end of previous epoch). '
                             'Keeps executor.step and WarmupLR phase aligned; optimizer state is still re-created from yaml.')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    if deepspeed is not None:
        parser = deepspeed.add_config_arguments(parser)
    else:
        parser.add_argument('--deepspeed_config', default=None, help='deepspeed config (ignored when using torch_ddp)')
    args = parser.parse_args()
    return args

@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint (weights-only for torch_ddp checkpoints)
    model = configs[args.model]
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            model_state = model.state_dict()
            filtered_ckpt = {}
            skipped = []
            # Some checkpoints may store a dict like {"model": state_dict, ...}
            state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_ckpt[k] = v
                elif k in model_state:
                    skipped.append(f'{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}')
            if skipped:
                logging.warning(f'Skipped {len(skipped)} shape-mismatched keys: {skipped}')
            model.load_state_dict(filtered_ckpt, strict=False)
            logging.info(f'Loaded {len(filtered_ckpt)}/{len(state_dict)} keys from checkpoint')
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # v15: initialize tokenizer_proj from codebook-embedding correspondence
    if hasattr(model, 'init_tokenizer_proj_from_codebook'):
        model.init_tokenizer_proj_from_codebook()

    # Differential learning rate: emotion modules get higher lr, backbone gets very low lr
    emotion_prefixes = (
        'emo_mlp', 'emo_proj', 'emo_VAD_inten_proj', 'emosty_layer_norm',
        'emotion_cross_attn', 'emotion_to_speech_proj', 'emotion_classifier',
        'map_speaker_embedding', 'spk_embed_affine_layer', 'spk_embed_proj',
        'sty_proj', 'azimuth_emb', 'elevation_emb',
    )
    model._emotion_prefixes = emotion_prefixes

    # L2 anchor: store init weights as reference to prevent catastrophic forgetting
    anchor_weight = configs.get('train_conf', {}).get('anchor_weight', 0)
    if anchor_weight > 0:
        anchor_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                anchor_dict[name] = param.data.clone()
        model._anchor_weights = anchor_dict
        model._anchor_weight = anchor_weight
        logging.info(f'[anchor] L2 anchor enabled: weight={anchor_weight}, {len(anchor_dict)} params anchored')

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    save_model(model, 'init', info_dict)

    # Get executor
    executor = Executor(gan=gan)
    resume_train_step = getattr(args, 'resume_train_step', None)
    if resume_train_step is not None:
        if gan:
            logging.warning('--resume_train_step is ignored for GAN training')
        elif scheduler is None:
            logging.warning('--resume_train_step set but scheduler is None; ignoring')
        else:
            rs = int(resume_train_step)
            if rs < 0:
                raise ValueError(f'--resume_train_step must be >= 0, got {rs}')
            executor.step = rs
            sched_last = max(rs - 1, -1)
            if hasattr(scheduler, 'set_step'):
                scheduler.set_step(sched_last)
            else:
                scheduler.last_epoch = sched_last
            lrs = scheduler.get_lr()
            for param_group, lr in zip(optimizer.param_groups, lrs):
                param_group['lr'] = lr
            logging.info('Resume train step: executor.step=%s, scheduler last_epoch=%s, lr[0]=%s',
                         executor.step, getattr(scheduler, 'last_epoch', '?'), optimizer.param_groups[0]['lr'])

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Start training loop
    start_epoch = int(getattr(args, 'start_epoch', 0) or 0)
    if start_epoch < 0:
        raise ValueError(f'--start_epoch must be >= 0, got {start_epoch}')
    for epoch in range(start_epoch, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)

if __name__ == '__main__':
    main()
