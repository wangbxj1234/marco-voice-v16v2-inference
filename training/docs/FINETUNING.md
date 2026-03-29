# Flow 微调与训练（完全在本仓库内）

## 环境

```bash
cd /path/to/marco-voice-v16v2-inference
source training/path.sh
pip install -r requirements.txt
```

需要 **CUDA + NCCL**（多卡）或单卡 `CUDA_VISIBLE_DEVICES=0`。训练依赖 **DeepSpeed**（已写在根目录 `requirements.txt`）。

## 公开超参快照（约 epoch 199）

标量字段见 **`training/reference_public_epoch199_snapshot.yaml`**（与内部 `epoch_199_whole.yaml` 一致，已去掉路径与张量垃圾）。

摘要：

- **阶段 1**：`training/conf/cosyvoice_emosphere_v16_v2.yaml`，Adam 峰值 **1e-3**，**WarmupLR**，**warmup_steps=93**，**max_epoch=200**，**grad_clip=5**。
- **阶段 2（续训降峰）**：`training/conf/cosyvoice_emosphere_v16_v2_resume_lowlr.yaml`，峰值 **2e-4**，**warmup_steps 仍为 93**；从 epoch *K* 继续时，应用上一 epoch 的 `epoch_{K-1}_whole.yaml` 里的 **`step:`** 作为 `train.py --resume_train_step`，避免 LR 相位错位。
- **epoch 199 末**：日志中 **lr ≈ 2.48×10⁻⁵**，**global step = 6070**（示例；你本地以 yaml 为准）。

## 数据：因果 S3 token

1. 准备 Emosphere 格式目录 `SRC_DIR`（`wav.scp`、`text`、`utt2spk`、各类 `*.pt` 嵌入，与你们现有数据一致）。

2. 使用仓库内脚本（与推理同一套 mel + `S3TokenizerV1`）：

```bash
source training/path.sh
python training/tools/extract_speech_token_s3.py \
  --dir "$SRC_DIR" \
  --tokenizer_pt /path/to/s3_tokenizer.pt \
  --device cuda \
  --batch_size 32
```

3. 生成 parquet：

```bash
export SRC_DIR=...
export DES_DIR=...
bash training/scripts/prep_parquet_s3causal.sh
```

**一键最小示例**（单条 utterance，用于验证流水线）：

```bash
export TOKENIZER_PT=weights/s3_tokenizer.pt
bash training/scripts/prepare_smoke_parquet.sh
```

4. 若已有 parquet 仅缺/需重算 `speech_token` 列：

```bash
python training/tools/retokenize_parquet_s3.py \
  --in_list OLD/data.list \
  --out_dir NEW/shards \
  --out_list NEW/data.list \
  --tokenizer_pt weights/s3_tokenizer.pt
```

## 训练命令

```bash
source training/path.sh
export MARCO_V16V2_DATA_LIST=/path/to/data.list
export FLOW_INIT_CKPT=/path/to/flow.pt
export CUDA_VISIBLE_DEVICES=0,1
bash training/scripts/run_train_flow_v16v2.sh
```

常用环境变量：

- `MARCO_V16V2_CONFIG` — 默认 `training/conf/cosyvoice_emosphere_v16_v2.yaml`
- `MARCO_V16V2_EXP` — 实验子目录名
- `MARCO_RESUME_CKPT` / `MARCO_START_EPOCH` / `MARCO_RESUME_TRAIN_STEP` — 续训
- `SKIP_POST_AVERAGE=1` — 跳过 `average_model.py`（调试用）

续训降 LR：`bash training/scripts/run_resume_lowlr_after_plateau.sh`（需设置 `MARCO_RESUME_CKPT`、`PREV_YAML`、`MARCO_V16V2_DATA_LIST`）。

## 推理衔接

新训练的 flow 可覆盖 `weights/flow.pt` 或用 `infer.py --flow_ckpt` 指向新 checkpoint。
