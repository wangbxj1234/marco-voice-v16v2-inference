# Marco Voice — v16 v2（独立仓库：推理 + 微调）

本仓库 **自包含**：因果 S3（1024 @ 25 Hz）+ Emosphere flow 的 **推理**、**数据准备**、**flow 训练** 全部在仓库内完成，**不要求**再 clone Marco-Voice、ft_cosy 或 CosyVoice 其他目录。

**预训练权重已发布**：全部大文件已上传至 Hugging Face 仓库 [**`wbxlala/marcov16v2`**](https://huggingface.co/wbxlala/marcov16v2)（`flow.pt`、`hift.pt`、`campplus.onnx`、因果 `s3_tokenizer.pt` 等）。可直接从该仓库下载到 `./weights/`，或仍用下文 manifest / 自建链接。

**与 Marco-Voice 主仓同机时**：请先阅读主仓根目录的 **`WORK_SUMMARY.md`**（路径示例：`Marco-Voice-main/WORK_SUMMARY.md`）。其中 **「补充：v16 v2 起点修正」「低学习率续训」「v16 v2 重建测评工作流」** 等节写明了：因果 **`TOKENIZER_PT` 必须与建 `parquet_s3causal25hz` 时一致**、`FLOW_INIT_CKPT` / `epoch_158_whole.pt` 来源、`data.list` 位置、rerun158 实验名与推理用 `flow.pt` 关系等。**不要只读本 README 就猜路径。** 本仓库脚本与主仓 `Training/*.sh` 的对照、软链复用 `.pt` 见 **`training/docs/MARCO_TRAINING_PARITY.md`**。

---

## 0. 环境（任选其一）

### A. 已有 CUDA PyTorch（推荐与训练时一致）

```bash
cd marco-voice-v16v2-inference
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
# 按你的 CUDA 版本从 https://pytorch.org 安装 torch / torchaudio 后：
pip install -r requirements.txt
```

### B. 与 Marco-Voice 同机、使用其 `marco` 环境（可选）

若本目录在 Marco-Voice 旁且存在 `../marco/bin/python`，可直接：

```bash
export MARCO_PYTHON=/path/to/Marco-Voice/marco/bin/python
# 下文凡 `python` 均可换为 `"$MARCO_PYTHON"`
```

`verify.sh` 在未设置 `PYTHON` 时会自动探测 `../marco/bin/python`。

---

## 1. 下载权重（推理 + 训练起点）

### 1.1 从 Hugging Face 拉取（推荐）

官方权重仓库：**[https://huggingface.co/wbxlala/marcov16v2](https://huggingface.co/wbxlala/marcov16v2)**

任选其一：

```bash
# 需已安装 huggingface-cli（pip install huggingface_hub）
huggingface-cli download wbxlala/marcov16v2 --local-dir weights
```

或网页进入上述仓库，将所需文件下载到本地 `weights/` 目录。

### 1.2 使用 manifest 脚本

若你使用自建镜像或零散链接：

```bash
cp weights_manifest.example.json weights_manifest.json
# 编辑 weights_manifest.json：为 flow.pt / hift.pt / campplus.onnx / s3_tokenizer.pt 填入 https:// 或 hf:org/repo:file 链接
python scripts/download_weights.py --manifest weights_manifest.json --output-dir weights
```

完成后应存在：

- `weights/flow.pt` `weights/hift.pt` `weights/campplus.onnx`
- `weights/cosyvoice.yaml`（脚本从 `configs/cosyvoice.yaml` 复制）
- `weights/s3_tokenizer.pt`（因果 S3 导出，与训练 parquet 必须使用**同一**导出）

---

## 2. 推理（复制即可运行）

**用哪个 `python`**：在 §0 创建的虚拟环境里先 `source .venv/bin/activate`，下面命令里的 `python` 即该环境；或与 Marco-Voice 同目录树时用固定路径，例如  
`/path/to/Marco-Voice/marco/bin/python infer.py ...`（`verify.sh` 会自动探测 `../marco/bin/python`）。

**示例输入**：仓库自带 **`sample_inputs/esd_source_spk0002_neutral_u000282_long.wav`**（ESD 长句，与 flow streaming 复现脚本默认一致）。未传 `--source_wav` 时与 `--prompt_wav` 相同，即自重建。

```bash
cd marco-voice-v16v2-inference
source training/path.sh
python infer.py \
  --weights_dir weights \
  --tokenizer_pt weights/s3_tokenizer.pt \
  --prompt_wav sample_inputs/esd_source_spk0002_neutral_u000282_long.wav \
  --out_wav outputs/demo.wav
```

首次运行会从 ModelScope / Hugging Face **自动拉取** emotion2vec 与 wav2vec（需联网），属正常现象。读 wav / 写 wav 走 **torchaudio**；若解码报错请本机安装 **FFmpeg**（如 `apt-get install -y ffmpeg`）。

自检（不加载大权重）：

```bash
python infer.py --smoke_imports
bash verify.sh
```

### 2.1 复现 Flow Streaming v2（epoch 92，hop=8）

我们将 flow streaming v2 的 `epoch_92_whole.pt`（CV 最优候选）公开在：

- [ft_flowv2_epoch_92_whole.pt (Google Drive)](https://drive.google.com/file/d/1F4upBZ0mX6BKLO1S2dF3lrd7VyvP35sA/view?usp=drive_link)

下载后可直接运行复现脚本（默认做 long source 自重建，对比 stream/offline，并额外输出 baseline 对照）：

```bash
source training/path.sh
FLOW_CKPT=/abs/path/to/ft_flowv2_epoch_92_whole.pt \
  bash training/scripts/reproduce_flow_stream_hop8.sh
```

默认关键参数：

- `HOP_TOKENS=8`
- `FLOW_TIMESTEPS=8`
- `PROMPT_WAV=SOURCE_WAV=sample_inputs/esd_source_spk0002_neutral_u000282_long.wav`
- 输出目录：`outputs/flow_stream_ep92_demo/`

可覆盖示例：

```bash
FLOW_CKPT=/abs/path/to/ft_flowv2_epoch_92_whole.pt \
PROMPT_WAV=sample_inputs/esd_prompt_spk0001_neutral_u000001.wav \
SOURCE_WAV=sample_inputs/esd_source_spk0002_neutral_u000282_long.wav \
RUN_BASELINE=0 \
bash training/scripts/reproduce_flow_stream_hop8.sh
```

---

## 3. 训练 / 微调（全流程在仓库内）

### 3.1 准备自己的 Emosphere 目录

目录内需有：`wav.scp`、`text`、`utt2spk`、`utt2embedding.pt`、`spk2embedding.pt`、`utt_emo.pt`、`emotion_embedding.pt`、`low_level_embedding.pt`（与主项目 Emosphere 数据格式一致）。

对目录 `SRC_DIR` 抽取因果 token 并写 `utt2speech_token.pt`：

```bash
source training/path.sh
python training/tools/extract_speech_token_s3.py \
  --dir "$SRC_DIR" \
  --tokenizer_pt weights/s3_tokenizer.pt \
  --device cuda \
  --batch_size 32
```

生成 parquet + `data.list`：

```bash
export SRC_DIR=/path/to/your_processed_train
export DES_DIR=/path/to/parquet_out
bash training/scripts/prep_parquet_s3causal.sh
# 训练时使用：MARCO_V16V2_DATA_LIST=$DES_DIR/data.list
```

若只有带 `audio_data` 的旧 parquet，可用仓库内工具重算 `speech_token`：

```bash
python training/tools/retokenize_parquet_s3.py \
  --in_list /path/to/old/data.list \
  --out_dir /path/to/new_shards \
  --out_list /path/to/new/data.list \
  --tokenizer_pt weights/s3_tokenizer.pt
```

完整说明与公开超参（含 epoch≈199 学习率快照）：**`training/docs/FINETUNING.md`**

### 3.2 启动 flow 训练

```bash
source training/path.sh
export MARCO_V16V2_DATA_LIST=/path/to/parquet_out/data.list
export FLOW_INIT_CKPT=weights/flow.pt    # 或你自己的 checkpoint
export CUDA_VISIBLE_DEVICES=0,1
bash training/scripts/run_train_flow_v16v2.sh
```

平台期后降低峰值学习率续训：

```bash
export MARCO_RESUME_CKPT=exp/.../epoch_129_whole.pt
export PREV_YAML=exp/.../epoch_129_whole.yaml
export MARCO_V16V2_DATA_LIST=...
bash training/scripts/run_resume_lowlr_after_plateau.sh
```

---

## 4. 一键验证（推理 + smoke 数据 + 1 epoch 训练）

在已安装依赖、有 **GPU** 的机器上，用本地权重目录与因果 tokenizer 路径：

```bash
source training/path.sh
export TOKENIZER_PT=/abs/path/to/s3_tokenizer.pt    # 或 weights/s3_tokenizer.pt
export WEIGHTS_DIR=/abs/path/to/dir_with_flow_hift_campplus
bash scripts/verify_end_to_end.sh
```

该脚本会：复制权重到 `.verify_runtime_weights/` → 跑 `infer.py` → 用内置 `sample_inputs` 建最小 parquet → 用 `training/conf/cosyvoice_emosphere_v16_v2_smoke1.yaml`（`max_epoch: 1`）跑 **单卡** flow 训练并跳过 checkpoint 平均。

**已在以下环境跑通（exit 0）**：`WEIGHTS_DIR=Marco-Voice/pretrained_models/marco_voice/cosyvoice_emosphere_v16_v2`（`cp -L` 解析 symlink）+ 与训练一致的因果 `s3_tokenizer` 导出 `.pt`。

---

## 5. 仓库结构（与复现相关）

| 路径 | 作用 |
|------|------|
| `infer.py` | 推理入口 |
| `configs/cosyvoice.yaml` | 推理用 flow/hift 结构（无 LLM 权重） |
| `training/conf/cosyvoice_emosphere_v16_v2*.yaml` | 训练用完整 HyperPyYAML（`--model flow`） |
| `training/tools/extract_speech_token_s3.py` | 写 `utt2speech_token.pt` |
| `training/tools/make_parquet_list_eposhere.py` | parquet + data.list |
| `training/tools/retokenize_parquet_s3.py` | 从 parquet 内音频重算 token |
| `training/scripts/run_train_flow_v16v2.sh` | DDP + DeepSpeed 训练 |
| `training/path.sh` | `source` 后设置 `PYTHONPATH` |
| `scripts/verify_end_to_end.sh` | 全栈验证 |

---

## 6. License

见 `LICENSE`、`NOTICE.md`。
