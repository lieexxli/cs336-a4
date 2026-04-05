# CS336 Assignment 4 复现指南

本仓库的工作流可以分成两条路线：

| 路线 | 目标 | 适合场景 | 典型产物 |
|---|---|---|---|
| 小样本跑通 | 先验证所有数据处理和分类器训练步骤都能在本地走通 | 本地磁盘有限，先做 smoke test | `artifacts/small/out/models/quality.bin`、`artifacts/small/data/leaderboard/classifier/quality.bin`、`artifacts/small/data/tokens.bin` |
| 全量复现 | 用更多 Common Crawl 数据重跑正式数据管线 | 准备真正做大规模过滤或最终训练 | 更大的 `artifacts/full/data/01-english` ~ `artifacts/full/data/04-classified`、正式版 `artifacts/full/data/tokens.bin` |

如果你想让人类或 AI 直接“一键按路线执行”，优先用这两个入口：

```bash
make -f Makefile.small all
make -f Makefile.full all
```

这两个入口现在会在终端里打印每个阶段的实际耗时，以及整条路线的总耗时。

查看可用分步 target：

```bash
make -f Makefile.small help
make -f Makefile.full help
```

### Make 快速上手

如果你准备用 `make` 驱动整条流程，最常用的是下面这些命令。

#### 1. 一键跑完整路线

```bash
make -f Makefile.small all
make -f Makefile.full all
```

- `small` 的最终产物在 `artifacts/small/...`
- `full` 的最终产物在 `artifacts/full/...`
- 两条路线共享 `data/` 下的外部资源和 Common Crawl 下载缓存

#### 2. 先看当前参数

```bash
make -f Makefile.small show-config
make -f Makefile.full show-config
```

如果你要改全量参数，也可以直接这样看：

```bash
make -f Makefile.full show-config FULL_WET_FILES=100 FULL_TRAIN_EXAMPLES=5000 FULL_VALID_EXAMPLES=200
```

#### 3. 分阶段单独执行

这两个 `Makefile` 都支持相同的一组阶段 target：

```bash
make -f Makefile.small common-prep
make -f Makefile.small download-cc
make -f Makefile.small quality-a
make -f Makefile.small pipeline-1-3
make -f Makefile.small quality-b
make -f Makefile.small pipeline-4-5
```

全量路线同理：

```bash
make -f Makefile.full common-prep
make -f Makefile.full download-cc
make -f Makefile.full quality-a
make -f Makefile.full pipeline-1-3
make -f Makefile.full quality-b
make -f Makefile.full pipeline-4-5
```

适合的场景：

- 只想先准备共享资源：跑 `common-prep`
- 只想确认 Common Crawl 下载/缓存逻辑：跑 `download-cc`
- Step 3 改完代码后只想从主 pipeline 中段继续：跑 `pipeline-1-3`
- 只想重训分类器 B：跑 `quality-b`

#### 4. 给 full 覆盖参数

参数覆盖可以加在任意 target 后面，不只 `all`：

```bash
make -f Makefile.full download-cc FULL_WET_FILES=100
make -f Makefile.full quality-b FULL_TRAIN_EXAMPLES=5000 FULL_VALID_EXAMPLES=200
make -f Makefile.full all FULL_WET_FILES=100 FULL_TRAIN_EXAMPLES=5000 FULL_VALID_EXAMPLES=200
```

#### 5. 清理某条路线自己的产物

```bash
make -f Makefile.small reset
make -f Makefile.full reset
```

`reset` 只会删除对应路线的 `artifacts/small` 或 `artifacts/full`，不会删除：

- `data/classifiers/`
- `data/wiki/`
- `data/paloma/`
- `data/commoncrawl/cache/`

#### 6. 查看完整帮助

```bash
make -f Makefile.small help
make -f Makefile.full help
```

#### 7. full 中断后怎么续跑

`Makefile.full` 的 Common Crawl 下载现在会把文件先写到共享缓存里的 `.part` 临时文件，下载并校验通过后再原子改名为正式缓存文件，并额外写一个 `.ok` 标记。

这意味着：

- 如果中断发生在 `download-cc` 阶段，重新执行同一个 target 会优先续传 `.part`
- 只有校验通过的缓存文件才会在后续 rerun 中被复用
- `artifacts/full/...` 里仍然只保存软链和中间产物，不会重复存一份大文件

这份文档按这个顺序组织：

1. 先看“整体结构”，知道代码、数据、模型分别放在哪里
2. 再看“全量所需硬件配置”，判断本机是否适合全量
3. 做一次“通用准备”
4. 然后二选一：
   - 只想验证流程：走“小样本跑通”
   - 想直接完整复现：直接走“全量复现”

所有命令默认都在仓库根目录执行。

---

## 1. 先看整体结构

### 1.1 你最后会得到什么

如果只做到“语言模型训练之前”为止，核心产物有 3 类：

| 产物 | 路径 | 作用 |
|---|---|---|
| 测试质量分类器 A | `artifacts/<route>/out/models/quality.bin` | 作业里的测试/辅助分类器，不参与最终主 pipeline |
| Leaderboard 质量分类器 B | `artifacts/<route>/data/leaderboard/classifier/quality.bin` | Step 4 实际使用的分类器 |
| 训练用 token 数据 | `artifacts/<route>/data/tokens.bin` | 最终语言模型训练输入 |

### 1.2 仓库结构与作用

下面这个结构只保留复现时最重要的目录：

```text
cs336-a4/
├── cs336_data/                       # 本作业自己的数据处理代码
│   ├── extract_text.py               # HTML -> 纯文本
│   ├── language_identification.py    # 语种识别
│   ├── harmful_content.py            # NSFW / 毒性分类
│   ├── gopher_quality_filters.py     # Gopher 规则过滤
│   ├── exact_deduplication.py        # Step 3 精确去重
│   ├── quality_classifier/           # 测试质量分类器 A 的训练与推理
│   └── leaderboard/                  # 主 pipeline（Step 1-5）与分类器 B
├── cs336-basics/                     # 最终语言模型训练代码
│   ├── configs/experiment/           # 训练配置
│   └── scripts/train.py              # 语言模型训练入口
├── tests/                            # 单元测试与夹具
├── data/                             # 共享输入、下载缓存与外部资源（不提交）
│   ├── classifiers/                  # 外部下载的 fastText 模型
│   ├── wiki/                         # Wikipedia 标题与 URL 列表
│   ├── commoncrawl/                  # CC 路径清单与下载缓存
│   ├── paloma/                       # Paloma validation.bin
│   └── .shared-stamps/               # 共享准备阶段的 stamp
├── artifacts/                        # make 路线的隔离产物根目录
│   ├── small/                        # 小样本路线全部中间结果、模型与 stamp
│   └── full/                         # 全量路线全部中间结果、模型与 stamp
└── USAGE.md                          # 本文档
```

### 1.3 代码、数据、模型分别是什么

| 类别 | 典型位置 | 作用 |
|---|---|---|
| 代码 | `cs336_data/`, `cs336-basics/` | 真正执行过滤、采样、训练 |
| 共享输入与缓存 | `data/classifiers/`, `data/wiki/`, `data/paloma/`, `data/commoncrawl/cache/` | 两条路线都能安全复用的外部资源 |
| 路线中间产物 | `artifacts/<route>/data/01-english` ~ `artifacts/<route>/data/04-classified` | 某条路线自己的 Step 1-4 输出 |
| 路线模型文件 | `artifacts/<route>/out/models/quality.bin`, `artifacts/<route>/data/leaderboard/classifier/quality.bin` | 某条路线训练出的 fastText 模型 |
| 最终训练输入 | `artifacts/<route>/data/tokens.bin` | 给语言模型训练脚本使用的 tokenized 数据 |

### 1.4 人类和 AI 都该遵守的执行约定

为避免复现中途跑偏，建议统一按下面规则执行：

1. 默认所有命令都在仓库根目录运行，只有最终 LLM 训练时才显式 `cd cs336-basics`。
2. “小样本跑通”与“全量复现”是两条路线；不要把小样本中间结果误当成全量结果。
3. `data/`、`out/models/`、`artifacts/`、下载的 `.bin/.gz` 都是本地产物，不应提交到 Git。
4. 如果目标只是“做到 `tokens.bin` 为止”，执行到 Step 5 就停止，不要继续跑 LLM 训练。
5. `make` 路线下，`data/` 只放共享输入和下载缓存；`artifacts/small` 与 `artifacts/full` 各自保存自己的中间结果和模型。
6. 本文统一使用仓库内相对路径，如 `data/...`，不再额外介绍集群专用路径。
7. 每完成一个小节，先检查该节列出的“关键产物”是否已经生成，再进入下一节。
8. 如果你不想手动逐条敲命令，可以直接使用 `Makefile.small` 或 `Makefile.full`；下面的 shell 命令更多是给排障和分步执行用的。
9. 这两个 `Makefile` 不再共用中间产物目录；它们只共享 `data/` 下的外部资源和 Common Crawl 下载缓存。

---

## 2. 全量所需硬件配置

如果你的目标是“做到 `tokens.bin` 为止，不做最后语言模型训练”，那么：

- **GPU 不是刚需**
- **CPU、内存、磁盘才是刚需**

推荐按下面几档理解：

| 场景 | CPU | 内存 | 磁盘 | GPU | 备注 |
|---|---:|---:|---:|---:|---|
| 小样本跑通 | 8-16 核 | 32-64 GB | 100-300 GB SSD | 不需要 | 用来验证所有步骤都能执行 |
| 全量训练前 pipeline 最低可用 | 32 核 | 128 GB | 2 TB NVMe | 不需要 | 能做，但 Step 3 内存压力偏大 |
| 全量训练前 pipeline 推荐单机 | 48-64 核 | 256 GB | 4 TB NVMe | 不需要 | 更稳妥，适合保留更多中间产物 |
| 如果连最后 LM 训练也要做 | 48-64 核 | 256 GB | 4 TB NVMe | `2 x 40GB+` | 更接近默认训练脚本配置 |

### 2.1 为什么全量主要吃 CPU / RAM / SSD

- Step 1 语种过滤、Step 2 启发式过滤、Step 4 分类器过滤、Step 5 tokenize，基本都是 CPU 任务。
- 两个 fastText 质量分类器训练也主要吃 CPU，不依赖 GPU。
- 真正最容易成为瓶颈的是 Step 3 `exact_dedupe`：
  - 当前实现会把大量行哈希放进 Python `dict`
  - 这一步对 **内存** 最敏感
  - 也是全链路最不适合小内存机器的一步

### 2.2 磁盘为什么要至少 2 TB，推荐 4 TB

按文档默认“5,000 个 WET 文件”的量级，经验上：

| 阶段 | 大小估算 |
|---|---|
| 原始 WET 压缩文件 | ~1 TB |
| Step 1 后 | ~140 GB |
| Step 2-3 后 | ~40 GB |
| Step 4 后 | ~15 GB |
| 最终 `tokens.bin` | ~12.8 GB |

如果你愿意在每一步完成后主动删除前一阶段的中间产物：

- `2 TB NVMe` 可以作为下限

如果你希望同时保留：

- 原始 WET
- Step 1/2/3/4 中间结果
- 最终 `tokens.bin`

那更现实的是：

- `4 TB NVMe`

不建议用机械硬盘做这套流程的主存储。

### 2.3 如果以后还想做最终 LM 训练

这一步不属于本文核心，但为了避免低估资源，这里单独说明。

`cs336-basics` 当前默认训练配置大致是：

- 约 `162M` 参数
- `context_length = 512`
- `train_batch_size = 128`
- `train_steps = 100000`
- 默认脚本按 `2 GPU` 写

所以如果你还想继续往下做最终 LM 训练：

- 推荐 `2 x 40GB` 或更大显存的 GPU
- `2 x 24GB` 不是完全不能试，但通常需要你自己改 batch size 或训练参数

如果你只做到 `tokens.bin`，就不需要专门配 GPU。

### 2.4 如果只考虑训练环节，硬件要求会低很多

前面那套 `48-64` 核、`128-256GB` 内存、`2-4TB` NVMe，主要是给**全量数据处理 pipeline**准备的，尤其是 Step 3 精确去重。

如果你已经有了训练数据，只考虑训练本身：

| 训练内容 | CPU | 内存 | 磁盘 | GPU | 备注 |
|---|---:|---:|---:|---:|---|
| 训练测试分类器 A / Leaderboard 分类器 B | 8-16 核 | 16-32 GB | 20-50 GB | 不需要 | fastText 训练主要吃 CPU |
| 只训练最终语言模型，且 `tokens.bin` 已就绪 | 8-16 核 | 32-64 GB | 50-100 GB NVMe | `2 x 40GB+` 更稳 | GPU 是核心瓶颈 |

更准确地说：

- **全量数据准备**：吃 CPU / RAM / NVMe
- **模型训练**：主要吃 GPU；CPU、内存、硬盘要求反而没那么夸张

---

## 3. 通用准备

### 3.1 安装依赖

```bash
uv sync
```

### 3.2 下载 NLTK 分词数据

```bash
uv run python - <<'PY'
import nltk
nltk.download("punkt_tab")
PY
```

### 3.3 创建目录

```bash
mkdir -p \
  data/classifiers \
  data/wiki \
  data/commoncrawl \
  data/CC \
  data/raw \
  data/paloma \
  data/leaderboard \
  data/leaderboard/classifier \
  out/models
```

### 3.4 下载外部分类器模型

本文统一把外部分类器模型放在 `data/classifiers/`。

```bash
wget -O data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin \
  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"

wget -O data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin \
  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"

wget -O data/classifiers/lid.176.bin \
  "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
```

### 3.5 准备 Wikipedia 标题与 URL 列表

```bash
wget -O data/wiki/titles.gz \
  "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"

zcat data/wiki/titles.gz \
  | tail -n +2 \
  | awk '{print "https://en.wikipedia.org/wiki/"$1}' \
  | gzip > data/wiki/enwiki-20240420-extracted_urls.txt.gz
```

### 3.6 构建 Paloma `c4_100_domains` 验证集 `.bin`

访问 Paloma 前，先确保你已经登录 Hugging Face 并同意该数据集的访问协议。

```bash
uv run python - <<'PY'
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    ds = load_dataset("allenai/paloma", "c4_100_domains", split="validation")
except ValueError:
    ds = load_dataset("allenai/paloma", "c4_100_domains", split="val")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
all_ids = []
for item in tqdm(ds, desc="Tokenizing Paloma"):
    all_ids.extend(tokenizer.encode(item["text"]) + [tokenizer.eos_token_id])

os.makedirs("data/paloma", exist_ok=True)
out_path = "data/paloma/tokenized_paloma_c4_100_domains_validation.bin"
np.array(all_ids, dtype=np.uint16).tofile(out_path)
print(f"Saved {len(all_ids)} tokens to {out_path}")
PY

ln -sf ../paloma/tokenized_paloma_c4_100_domains_validation.bin \
  data/leaderboard/tokenized_paloma_c4_100_domains_validation.bin
```

仓库里的 `cs336-basics/configs/experiment/*.yaml` 现在已经默认指向项目内路径
`data/paloma/tokenized_paloma_c4_100_domains_validation.bin`。

---

## 4. 版本 A：小样本跑通

### 4.1 目标

这条路线的目标是：

- 跑通两个 fastText 质量分类器
- 跑通 Leaderboard Step 1 到 Step 5
- 生成一个小的 `artifacts/small/data/tokens.bin`
- 不做最终语言模型训练

建议磁盘预留：`5G` 到 `10G`

如果你只是想最快跑通整条小样本路线，直接执行：

```bash
make -f Makefile.small all
```

执行时会依次打印 `common-prep`、`download-cc`、`quality-a`、`pipeline-1-3`、`quality-b`、`pipeline-4-5` 的实际用时。

如果想分步执行或单独重跑某一段：

```bash
make -f Makefile.small help
```

### 4.2 下载少量 Common Crawl 样本

#### 下载 1 个 WARC 给测试分类器 A 做负例

```bash
wget -O data/commoncrawl/warc.paths.gz \
  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/warc.paths.gz"

first_warc_path=$(zcat data/commoncrawl/warc.paths.gz | head -1)
wget -O data/CC/example.warc.gz \
  "https://data.commoncrawl.org/${first_warc_path}"
```

#### 下载 2 个 WET 给主 pipeline 做 smoke test

```bash
wget -O data/commoncrawl/wet.paths.gz \
  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz"

zcat data/commoncrawl/wet.paths.gz | head -2 | while read -r wet_path; do
  wget -P data/raw "https://data.commoncrawl.org/${wet_path}"
done
```

### 4.3 训练测试质量分类器 A

这套分类器只用于测试和作业要求，本身不参与最终 Leaderboard 主 pipeline。

#### 为本地 smoke test 写一个稳定的 Wikipedia 正例列表

随机从几十万 Wikipedia URL 里直接抽样，容易碰到 429 或抽到很怪的页面标题。  
本地小样本更推荐直接用一组固定英文页面：

```bash
cat > data/wiki/subsampled_positive_urls.txt <<'EOF'
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Computer_science
https://en.wikipedia.org/wiki/Mathematics
https://en.wikipedia.org/wiki/Physics
https://en.wikipedia.org/wiki/Chemistry
https://en.wikipedia.org/wiki/Biology
https://en.wikipedia.org/wiki/History
https://en.wikipedia.org/wiki/Philosophy
https://en.wikipedia.org/wiki/Economics
https://en.wikipedia.org/wiki/Statistics
https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Deep_learning
https://en.wikipedia.org/wiki/Natural_language_processing
https://en.wikipedia.org/wiki/Neural_network
https://en.wikipedia.org/wiki/Algorithm
https://en.wikipedia.org/wiki/Data_science
https://en.wikipedia.org/wiki/Probability
https://en.wikipedia.org/wiki/Calculus
https://en.wikipedia.org/wiki/Linear_algebra
https://en.wikipedia.org/wiki/Quantum_mechanics
https://en.wikipedia.org/wiki/Relativity
https://en.wikipedia.org/wiki/Cell_(biology)
https://en.wikipedia.org/wiki/DNA
https://en.wikipedia.org/wiki/Evolution
https://en.wikipedia.org/wiki/World_War_II
https://en.wikipedia.org/wiki/Ancient_Greece
https://en.wikipedia.org/wiki/Renaissance
https://en.wikipedia.org/wiki/Democracy
https://en.wikipedia.org/wiki/Climate_change
https://en.wikipedia.org/wiki/Solar_System
EOF
```

#### 生成训练数据并训练

```bash
# Step A1: 下载这些 Wikipedia 页面，写成 WARC
bash cs336_data/quality_classifier/02-download_positive_urls.sh

# Step A2: 过滤正例
uv run python cs336_data/quality_classifier/03-filter_positive_samples.py

# Step A3: 从 WARC 中按正例数量采样负例
pos_n=$(wc -l < data/wiki/train_positive.txt)
uv run python cs336_data/quality_classifier/04-prepare_negative_samples.py \
  --warc-path data/CC/example.warc.gz \
  -n "${pos_n}"

# Step A4: 合并并切分
uv run python cs336_data/quality_classifier/05-merge_samples.py --n "${pos_n}"
uv run python cs336_data/quality_classifier/06-split_train_valid.py

# Step A5: 训练测试分类器 A
uv run python cs336_data/quality_classifier/07-train.py
```

关键产物：

- `data/wiki/quality.train`
- `data/wiki/quality.valid`
- `out/models/quality.bin`

如果你不是为了作业测试，而只是想跑通主 pipeline，这一节可以跳过。

### 4.4 运行 Leaderboard Step 1 到 Step 3

```bash
# Step 1: 语种过滤
uv run python cs336_data/leaderboard/01-language.py \
  --data-dir data/raw \
  --out-dir data/01-english \
  --single \
  --max-files 2

# Step 2: 启发式过滤
uv run python cs336_data/leaderboard/02-heuristics.py \
  --data-dir data/01-english \
  --out-dir data/02-heuristics \
  --single \
  --max-files 2

# Step 3: 精确去重
uv run python cs336_data/leaderboard/03-exact_dedupe.py \
  --data-dir data/02-heuristics \
  --outdir data/03-deduped
```

关键产物：

- `data/01-english`
- `data/02-heuristics`
- `data/03-deduped`

### 4.5 训练 Leaderboard 质量分类器 B

这是 Step 4 真正使用的分类器，和测试分类器 A 完全独立。

#### 小样本版本推荐参数

当你只有 2 个 WET 样本时，不要直接用默认 `28000/500`。  
推荐先用 `1000/100` 跑通：

```bash
# Step B1: 把 Paloma validation.bin 解码成文本
uv run python cs336_data/leaderboard/classifier/01-val_to_text.py

# Step B2: 采样正例
uv run python cs336_data/leaderboard/classifier/02-sample_positives.py \
  --num-train-examples 1000 \
  --num-valid-examples 100

# Step B3: 从 Step 3 输出里采样负例
uv run python cs336_data/leaderboard/classifier/03-select_negatives.py \
  --data-dir data/03-deduped \
  --num-train-examples 1000 \
  --num-valid-examples 100

# Step B4: 合并正负例
uv run python cs336_data/leaderboard/classifier/04-merge_samples.py

# Step B5: 训练 Leaderboard 分类器
uv run python cs336_data/leaderboard/classifier/05-train.py
```

关键产物：

- `data/leaderboard/classifier/quality.train`
- `data/leaderboard/classifier/quality.valid`
- `data/leaderboard/classifier/quality.bin`

### 4.6 运行 Leaderboard Step 4 到 Step 5

```bash
# Step 4: 用分类器 B 过滤
uv run python cs336_data/leaderboard/04-c4_100_classify.py \
  --data-dir data/03-deduped \
  --out-dir data/04-classified \
  --single \
  --max-files 2

# Step 5: Tokenize
uv run python cs336_data/leaderboard/05-tokenize.py \
  --input-dir data/04-classified \
  --output-path data/tokens.bin
```

关键产物：

- `data/04-classified`
- `data/tokens.bin`

`05-tokenize.py` 可能会打印 GPT-2 的“长度超过 1024”提示。  
这里只是在做 tokenizer 编码并写 `.bin`，这条提示不影响该步骤完成。

#### 小样本路线完成判定

当下面 3 个文件都存在时，可以认为“小样本路线已经跑通”：

- `out/models/quality.bin`
- `data/leaderboard/classifier/quality.bin`
- `data/tokens.bin`

---

## 5. 版本 B：全量复现

### 5.1 适用范围

这一节是**独立版全量流程**。

- 只依赖上面的“通用准备”（§ 3）
- **不依赖**先跑“小样本跑通”（§ 4）
- 如果你想直接从零做全量，可以直接从这里开始

如果你想最快按默认参数跑完整条全量路线，直接执行：

```bash
make -f Makefile.full all
```

执行时会打印每个阶段的实际用时，以及整条全量路线的总耗时。

默认参数可以按需覆盖，例如：

```bash
make -f Makefile.full all FULL_WET_FILES=100 FULL_TRAIN_EXAMPLES=5000 FULL_VALID_EXAMPLES=200
```

如果想查看完整 target 列表：

```bash
make -f Makefile.full help
```

`Makefile.full` 现在会在参数匹配时自动复用已有产物，避免不必要的重复计算：

- `common-prep` 会复用 `data/classifiers/`、`data/paloma/`、`data/wiki/` 下的共享资源
- `download-cc` 会复用 `data/commoncrawl/cache/` 里已经下载过的 WARC / WET，然后只在 `artifacts/full/data/CC` 和 `artifacts/full/data/raw` 里重建软链
- `quality-a` 只会在 `artifacts/full/data/wiki/` 和 `artifacts/full/out/models/` 已匹配当前参数时复用
- `pipeline-1-3` 只会在 `artifacts/full/data/03-deduped/meta.json` 里的 `total_files` 等于 `FULL_WET_FILES` 时复用
- `quality-b` 只会在 `artifacts/full/data/leaderboard/classifier/` 的样本数与当前参数一致时复用
- `pipeline-4-5` 只会在 `artifacts/full/data/04-classified` 文件数与 `FULL_WET_FILES` 一致，且 `artifacts/full/data/tokens.bin` 已存在时复用

这也意味着：

- `artifacts/small` 里的中间结果不会被 `Makefile.full` 误复用
- 但如果你显式把 `FULL_*` 参数改成与当前 `artifacts/full` 一致，`Makefile.full` 会直接跳过这些阶段

### 5.2 下载全量所需原始数据

#### 下载 1 个 WARC 供测试分类器 A 使用

测试分类器 A 只是作业里的辅助分类器，不参与最终主 pipeline。  
它的负例只需要少量原始 HTML，所以通常 1 个 WARC 就够了。

```bash
mkdir -p data/CC data/commoncrawl

wget -O data/commoncrawl/warc.paths.gz \
  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/warc.paths.gz"

first_warc_path=$(zcat data/commoncrawl/warc.paths.gz | head -1)
wget -O data/CC/example.warc.gz \
  "https://data.commoncrawl.org/${first_warc_path}"
```

#### 下载更多 WET 供主 pipeline 使用

把下面的 `N` 改成你要处理的 WET 文件数。  
如果你想对齐文档里的“全量量级”，就把 `N` 设成 `5000`。

```bash
mkdir -p data/raw data/commoncrawl

wget -O data/commoncrawl/wet.paths.gz \
  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz"

N=5000
zcat data/commoncrawl/wet.paths.gz | head -n "${N}" | while read -r wet_path; do
  wget -P data/raw "https://data.commoncrawl.org/${wet_path}"
done
```

### 5.3 从零训练测试质量分类器 A

这套分类器不参与最终主 pipeline，但如果你想把“语言模型训练之前的所有步骤”都做全，这一节也应完成。

#### Step A1: 从 Wikipedia URL 列表中随机采样正例页面

```bash
uv run python cs336_data/quality_classifier/01-sample_positive_urls.py \
  --inpath data/wiki/enwiki-20240420-extracted_urls.txt.gz \
  --outpath data/wiki/subsampled_positive_urls.txt \
  --max-urls 100000
```

#### Step A2: 下载采样到的 Wikipedia 页面

```bash
bash cs336_data/quality_classifier/02-download_positive_urls.sh
```

#### Step A3: 过滤正例

```bash
uv run python cs336_data/quality_classifier/03-filter_positive_samples.py
```

#### Step A4: 用 WARC 随机采样负例

```bash
pos_n=$(wc -l < data/wiki/train_positive.txt)
uv run python cs336_data/quality_classifier/04-prepare_negative_samples.py \
  --warc-path data/CC/example.warc.gz \
  -n "${pos_n}"
```

#### Step A5: 合并、切分、训练

```bash
uv run python cs336_data/quality_classifier/05-merge_samples.py --n "${pos_n}"
uv run python cs336_data/quality_classifier/06-split_train_valid.py
uv run python cs336_data/quality_classifier/07-train.py
```

关键产物：

- `data/wiki/quality.train`
- `data/wiki/quality.valid`
- `out/models/quality.bin`

### 5.4 全量运行 Leaderboard Step 1 到 Step 3

本地单机推荐显式加 `--mp`：

```bash
uv run python cs336_data/leaderboard/01-language.py \
  --data-dir data/raw \
  --out-dir data/01-english \
  --mp

uv run python cs336_data/leaderboard/02-heuristics.py \
  --data-dir data/01-english \
  --out-dir data/02-heuristics \
  --mp

uv run python cs336_data/leaderboard/03-exact_dedupe.py \
  --data-dir data/02-heuristics \
  --outdir data/03-deduped
```

关键产物：

- `data/01-english`
- `data/02-heuristics`
- `data/03-deduped`

### 5.5 从零训练 Leaderboard 质量分类器 B

这是 Step 4 实际使用的分类器，也是全量版本里必须重新训练的一步。

```bash
# Step B1: 把 Paloma validation.bin 解码成文本
uv run python cs336_data/leaderboard/classifier/01-val_to_text.py

# Step B2: 从 Paloma 中采样正例（默认 28,000 train + 500 valid）
uv run python cs336_data/leaderboard/classifier/02-sample_positives.py

# Step B3: 从 Step 3 的输出中采样负例
uv run python cs336_data/leaderboard/classifier/03-select_negatives.py \
  --data-dir data/03-deduped

# Step B4: 合并成 quality.train / quality.valid
uv run python cs336_data/leaderboard/classifier/04-merge_samples.py

# Step B5: 训练分类器
uv run python cs336_data/leaderboard/classifier/05-train.py
```

默认规模：

- 训练正例 `28000`
- 验证正例 `500`
- 训练负例 `28000`
- 验证负例 `500`

关键产物：

- `data/leaderboard/classifier/quality.train`
- `data/leaderboard/classifier/quality.valid`
- `data/leaderboard/classifier/quality.bin`

### 5.6 全量运行 Leaderboard Step 4 到 Step 5

本地单机推荐显式加 `--mp`：

```bash
uv run python cs336_data/leaderboard/04-c4_100_classify.py \
  --data-dir data/03-deduped \
  --out-dir data/04-classified \
  --mp

uv run python cs336_data/leaderboard/05-tokenize.py \
  --input-dir data/04-classified \
  --output-path data/tokens.bin
```

做到这里，就已经完成了“最终语言模型训练之前”的全部步骤。

#### 全量路线完成判定

当下面这些产物都存在时，可以认为“全量训练前流程已经完成”：

- `data/03-deduped`
- `data/leaderboard/classifier/quality.bin`
- `data/04-classified`
- `data/tokens.bin`

如果你也把测试质量分类器 A 一并做了，那么还应额外看到：

- `out/models/quality.bin`

---

## 6. 最终产物速查

下面这张表列的是“手动逐条执行 shell 命令”时的标准路径。  
如果你走 `Makefile.small` / `Makefile.full`，把前缀替换成 `artifacts/small/` 或 `artifacts/full/` 即可。

| 产物 | 说明 |
|---|---|
| `out/models/quality.bin` | 测试质量分类器 A |
| `data/leaderboard/classifier/quality.bin` | Leaderboard Step 4 实际使用的分类器 B |
| `data/01-english` | Step 1 输出 |
| `data/02-heuristics` | Step 2 输出 |
| `data/03-deduped` | Step 3 输出 |
| `data/04-classified` | Step 4 输出 |
| `data/tokens.bin` | Step 5 输出，可直接供最终训练使用 |

---

## 7. 如果你还要继续做最终语言模型训练

这一步不属于“训练前数据准备”，但命令放在这里方便对照。

### 本地多卡

```bash
cd cs336-basics
uv run torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config-name=experiment/bucketed.yaml
```

---

## 8. 测试与提交

### 运行测试

```bash
uv run pytest tests/ -v
```

如果只想测局部：

```bash
uv run pytest tests/test_pii.py -v
uv run pytest tests/test_quality.py -v
uv run pytest tests/test_deduplication.py -v
```

### 打包提交

```bash
bash test_and_make_submission.sh
```

---

## 9. 如果你先做过小样本，可复用什么

这一节只是给“先跑过小样本、现在想升级到全量”的读者看的。  
它不是全量流程的前置条件。  
如果你使用 `Makefile.small` / `Makefile.full`，那么 `artifacts/small` 与 `artifacts/full` 本来就是隔离的；全量路线只会复用下面这些共享输入和下载缓存。

| 产物 | 全量时能否直接复用 | 说明 |
|---|---|---|
| `uv sync` 安装好的环境 | 可以 | 直接复用 |
| `nltk` 的 `punkt_tab` | 可以 | 直接复用 |
| `data/classifiers/*.bin` | 可以 | NSFW、毒性、语种识别模型都可直接复用 |
| `data/wiki/titles.gz` | 可以 | 只是原始标题 dump |
| `data/wiki/enwiki-20240420-extracted_urls.txt.gz` | 可以 | 只是 Wikipedia URL 列表 |
| `data/paloma/tokenized_paloma_c4_100_domains_validation.bin` | 可以 | Leaderboard 分类器正例来源，可直接复用 |
| `data/leaderboard/tokenized_paloma_c4_100_domains_validation.bin` | 可以 | 只是兼容软链，可直接复用 |
| `data/commoncrawl/cache/CC/example.warc.gz` | 可以，但只够测试分类器 A | 这 1 个 WARC 只是共享下载缓存 |
| `data/commoncrawl/cache/raw/*.warc.wet.gz`（小样本已下载部分） | 可以部分复用 | `Makefile.full` 会优先复用已缓存的 WET，再继续补下载 |
| `artifacts/small/data/01-english`、`artifacts/small/data/02-heuristics`、`artifacts/small/data/03-deduped`、`artifacts/small/data/04-classified` | 不会复用 | 两条路线物理隔离，full 不会读取 small 的中间结果 |
| `artifacts/small/data/leaderboard/classifier/quality.bin` | 不会复用 | full 会只看 `artifacts/full/...` 下自己的分类器产物 |
| `artifacts/small/data/tokens.bin` | 不会复用 | full 会重新生成自己的 `artifacts/full/data/tokens.bin` |
