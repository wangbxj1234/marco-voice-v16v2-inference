# 

#!/usr/bin/env python3
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
import argparse
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import time
import torch


def _for_parquet_embedding(x):
    """Parquet/Arrow cannot store torch.Tensor; normalize to numpy or Python types."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _for_parquet_speech_token(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file, emo2parquet_file, lowlevelemo2parquet_file, uttemo2parquet_file):
    start_time = time.time()
    data_list = []
    try:
        for utt in tqdm(utt_list):
            data = open(utt2wav[utt], 'rb').read()
            data_list.append(data)

        wav_list = [utt2wav[utt] for utt in utt_list]
        text_list = [utt2text[utt] for utt in utt_list]
        spk_list = [utt2spk[utt] for utt in utt_list]
        uttembedding_list = [_for_parquet_embedding(utt2embedding[utt]) for utt in utt_list]
        spkembedding_list = [_for_parquet_embedding(spk2embedding[utt2spk[utt]]) for utt in utt_list]
        speech_token_list = [_for_parquet_speech_token(utt2speech_token[utt]) for utt in utt_list]
        utt_emo_list = [utt_emo[utt] for utt in utt_list]
        emotion_embedding_list = [_for_parquet_embedding(emotion_embedding[utt]) for utt in utt_list]
        low_level_emotion_embedding_list = [
            _for_parquet_embedding(low_level_emotion_embedding[utt]) for utt in utt_list
        ]

        # 保存到parquet,utt2parquet_file,spk2parquet_file, emo2parquet_file
        df = pd.DataFrame()
        df['utt'] = utt_list
        df['wav'] = wav_list
        df['audio_data'] = data_list
        df['text'] = text_list
        df['spk'] = spk_list
        df['utt_embedding'] = uttembedding_list
        df['spk_embedding'] = spkembedding_list
        df['speech_token'] = speech_token_list
        df['utt_emo'] = utt_emo_list
        # 新增情感嵌入列 [修改点2]
        df['emotion_embedding'] = emotion_embedding_list
        df['low_level_emotion_embedding'] = low_level_emotion_embedding_list
        df.to_parquet(parquet_file)
        with open(utt2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
        with open(emo2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
        with open(lowlevelemo2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
        with open(spk2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
        with open(uttemo2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in list(set(utt_list))}, f, ensure_ascii=False, indent=2)
        logging.info('spend time {}'.format(time.time() - start_time))
    except Exception as e:
        logging.error(e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    utt_emo = torch.load('{}/utt_emo.pt'.format(args.src_dir))
    emotion_embedding = torch.load('{}/emotion_embedding.pt'.format(args.src_dir))  # 原emotion_emobedding.pt改为emotion_embedding.pt
    low_level_emotion_embedding = torch.load('{}/low_level_embedding.pt'.format(args.src_dir)) 
    utts = list(utt2wav.keys())

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list, emo2parquet_list, lowlevelemo2parquet_list, uttemo2parquet_list = [], [], [], [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        emo2parquet_file = os.path.join(args.des_dir, 'emo2parquet_{:09d}.json'.format(i))
        uttemo2parquet_file = os.path.join(args.des_dir, 'uttemo2parquet_{:09d}.json'.format(i))
        lowlevelemo2parquet_file = os.path.join(args.des_dir, 'lowlevelemo2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        emo2parquet_list.append(emo2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        uttemo2parquet_list.append(uttemo2parquet_file)
        lowlevelemo2parquet_list.append(lowlevelemo2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file, emo2parquet_file, lowlevelemo2parquet_file, uttemo2parquet_file))
    pool.close()
    pool.join()
    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3,\
            open('{}/emo2data.list'.format(args.des_dir), 'w', encoding='utf8') as f4,\
            open('{}/lowlevelemo2data.list'.format(args.des_dir), 'w', encoding='utf8') as f5,\
            open('{}/uttemo2data.list'.format(args.des_dir), 'w', encoding='utf8') as f6:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
        for name in emo2parquet_list:
            f4.write(name + '\n')
        for name in lowlevelemo2parquet_list:
            f5.write(name + '\n')
        for name in uttemo2parquet_list:
            f6.write(name + '\n')