# CS336 Assignment 4 使用说明

本项目实现了一套用于**过滤和清洗 Common Crawl 数据**的工具链，最终目标是训练高质量的语言模型。

---

## 目录结构

```
cs336-a4/
├── cs336_data/                  # 核心实现模块
│   ├── extract_text.py          # HTML → 纯文本提取
│   ├── language_identification.py  # 语种识别（fastText）
│   ├── mask_pii.py              # PII 脱敏（邮箱/电话/IP）
│   ├── harmful_content.py       # 有害内容分类（NSFW/毒性）
│   ├── gopher_quality_filters.py  # Gopher 质量过滤规则
│   ├── exact_deduplication.py   # 精确去重
│   ├── minhash_deduplication.py # MinHash 模糊去重
│   ├── quality_classifier/      # 质量分类器（自训练 fastText）
│   │   ├── quality_classifier.py   # 推理接口
│   │   ├── 01~07-*.py           # 训练质量分类器的完整流程
│   │   └── 07-train.py          # 训练入口
│   └── leaderboard/             # 完整数据处理 Pipeline
│       ├── 01-language.py       # Step 1: 语种过滤
│       ├── 02-heuristics.py     # Step 2: 启发式质量过滤（C4 + Gopher）
│       ├── 03-exact_dedupe.py   # Step 3: 精确去重
│       ├── 04-c4_100_classify.py  # Step 4: 质量分类器过滤
│       ├── 05-tokenize.py       # Step 5: Tokenization
│       └── train.sh             # SLURM 训练脚本
├── cs336-basics/                # 助教提供的模型训练代码（勿修改）
├── tests/                       # 单元测试
├── pyproject.toml               # 项目依赖
└── get_assets.sh                # 下载预训练分类器模型
```

---

## 环境依赖

### 系统要求

- Python ≥ 3.11
- Linux / macOS（`os.sched_getaffinity` 在 Windows 不可用）
- 建议使用 GPU 进行模型训练

### 安装

本项目使用 `uv` 管理依赖：

```bash
# 安装所有依赖（含 cs336-basics 子模块）
uv sync
```

主要 Python 依赖：

| 包 | 用途 |
|---|---|
| `fasttext` | 语种识别 + 质量/NSFW/毒性分类 |
| `resiliparse` | HTML 解析与文本提取 |
| `mmh3` | 高速哈希（去重） |
| `nltk` | 分词（Gopher 过滤器） |
| `fastwarc` | 读取 WARC/WET 格式文件 |
| `xopen` | 透明读取压缩文件（.gz） |
| `transformers` | GPT-2 Tokenizer（Pipeline Step 2/5 及 Leaderboard 分类器） |
| `torch` | 模型训练 |
| `wandb` | 训练日志 |
| `tldextract` | 域名提取（黑名单过滤） |

---

## 所需数据与模型文件

> ⚠️ **代码库不含数据或模型权重**，需要单独获取。

### 数据与模型文件完整清单

| # | 文件 / 数据集 | 用途 | 获取方式 |
|---|---|---|---|
| 1 | `dolma_fasttext_nsfw_jigsaw_model.bin` | NSFW 分类 | [`bash get_assets.sh`](https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw) |
| 2 | `dolma_fasttext_hatespeech_jigsaw_model.bin` | 毒性分类 | [`bash get_assets.sh`](https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech) |
| 3 | `lid.176.bin` | 语种识别 | [`dl.fbaipublicfiles.com/.../lid.176.bin`](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)（126 MB） |
| 4 | `enwiki-20240420-extracted_urls.txt.gz` | 测试用质量分类器正例 | 可由 Wikipedia 标题 dump 快速生成（见下方） |
| 5 | CC WARC 文件（含 HTML） | 测试用质量分类器负例 | [`data.commoncrawl.org`](https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/warc.paths.gz) |
| 6 | CC WET 文件（纯文本） | 主过滤 Pipeline 输入 | [`data.commoncrawl.org`](https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz) |
| 7 | **Paloma `c4_100_domains` 验证集** | Leaderboard 分类器正例 + 模型评估 | [huggingface.co/datasets/allenai/paloma](https://huggingface.co/datasets/allenai/paloma)（需登录并同意协议） |
| 8 | **Leaderboard 分类器** (`classifier/quality.bin`) | Pipeline Step 4 实际分类器 | 需自行训练（见 § 6） |

### 1. 预训练 fastText 分类器（NSFW + 毒性）

代码依次查找以下路径（按优先级）：

1. `/data/classifiers/`（集群共享路径，优先）
2. `data/classifiers/`（**项目内推荐路径，个人服务器用这个**）
3. `../classifiers/`（项目根目录上一级，兼容旧路径）

**个人服务器推荐做法**（在项目根目录执行）：

```bash
mkdir -p data/classifiers
wget -O data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin \
    "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
wget -O data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin \
    "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"
```

> **注意**：`get_assets.sh` 是为集群环境设计的（优先从 `/data/classifiers/` 创建软链，否则下载到 `cs336_data/assets/`），**不适合个人服务器直接使用**，请用上述 wget 命令替代。

| 文件名 | 来源 |
|---|---|
| `dolma_fasttext_nsfw_jigsaw_model.bin` | [allenai/dolma-jigsaw-fasttext-bigrams-nsfw](https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw) |
| `dolma_fasttext_hatespeech_jigsaw_model.bin` | [allenai/dolma-jigsaw-fasttext-bigrams-hatespeech](https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech) |

### 2. 语种识别模型

下载 fastText 的语言识别模型（~126 MB），与 § 1 放在同一目录 `data/classifiers/`：

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O data/classifiers/lid.176.bin

# 压缩精简版（~917 KB，精度略降）
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -O data/classifiers/lid.176.ftz
```

- 官方文档：[fasttext.cc/docs/en/language-identification.html](https://fasttext.cc/docs/en/language-identification.html)

### 3. 质量分类器 A——测试用（`cs336_data/quality_classifier/`）

质量分类器是一个二分类 fastText 模型，用于区分"高质量文本"（维基百科风格）和"低质量文本"（普通 CC 网页），需要**自行准备训练数据并训练**。

#### 训练数据来源

| 类别 | 标签 | 来源 |
|---|---|---|
| 正例（高质量） | `__label__positive` | 维基百科页面（通过 Wikipedia URL 列表爬取） |
| 负例（低质量） | `__label__negative` | Common Crawl 原始 WARC 文件中随机采样的网页 |

#### 完整数据准备流程（7步）

**所需原始数据：**

**a. 维基百科 URL 列表（测试分类器正例）**

原集群提供的是预处理文件，为了本地平替，你可以直接下载官方的维基百科全量页面标题并拼接出完整的 URL：

```bash
mkdir -p data/wiki
# 1. 下载维基百科全量页面标题库（约 100 MB）
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz -O data/wiki/titles.gz

# 2. 将标题解压并拼接为真实的维基百科 URL 列表，再次压缩以符合输入要求 (.txt.gz)
zcat data/wiki/titles.gz | tail -n +2 | awk '{print "https://en.wikipedia.org/wiki/"$1}' | gzip > data/wiki/enwiki-20240420-extracted_urls.txt.gz
```
十几秒即可自动生成几十万个正例链接的压缩包，完美替代复杂的 SQL 库解析逻辑。

**b. CC WARC 文件（含 HTML）**

负例需要包含 HTML 的原始 WARC 文件（不是 WET）。

推荐先下载官方文件清单，再从清单里挑一个真实存在的文件保存为 `data/CC/example.warc.gz`。这样比把某个固定 segment 路径写死在文档里更稳妥：

```bash
mkdir -p data/CC data/commoncrawl

# 下载 WARC 文件列表
wget -O data/commoncrawl/warc.paths.gz \
    https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/warc.paths.gz
zcat data/commoncrawl/warc.paths.gz | head -3  # 预览前几个文件路径

# 下载单个 WARC 文件（主 Pipeline 进度和检验用）
# 这里把清单中的第一个文件保存为 data/CC/example.warc.gz，便于直接复用后续命令
first_warc_path=$(zcat data/commoncrawl/warc.paths.gz | head -1)
wget -O data/CC/example.warc.gz "https://data.commoncrawl.org/${first_warc_path}"
```

> 💡 本地只想先跑通 `04-prepare_negative_samples.py` 的话，下载 1 个 WARC 文件就够了；压缩后大约 1 GB 左右。

```bash
# Step 1：从维基百科 URL 列表中随机采样 10万个 URL
uv run python cs336_data/quality_classifier/01-sample_positive_urls.py \
    --inpath  data/wiki/enwiki-20240420-extracted_urls.txt.gz \
    --outpath data/wiki/subsampled_positive_urls.txt \
    --max-urls 100000

# Step 2：用 wget 批量爬取这些 URL，保存为 WARC 文件
#         （结果存为 data/wiki/unfiltered_positive_samples.warc.gz）
bash cs336_data/quality_classifier/02-download_positive_urls.sh

# Step 3：从爬取结果中过滤正例
#         过滤条件：英文 + 非NSFW + 非毒性 + 通过 Gopher 质量过滤
#         输出：data/wiki/train_positive.txt（每行一个 fastText 样本）
uv run python cs336_data/quality_classifier/03-filter_positive_samples.py

# Step 4：从 CC WARC 文件中随机采样负例
#         输出：data/wiki/train_negative.txt
uv run python cs336_data/quality_classifier/04-prepare_negative_samples.py \
    --warc-path data/CC/example.warc.gz \
    -n 13500   # 采样数量，与正例保持平衡

# Step 5：合并正负例，平衡类别数量
uv run python cs336_data/quality_classifier/05-merge_samples.py \
    --n 13500  # 正负各取 13500 条

# Step 6：切分训练集 / 验证集（默认 90% / 10%）
#         输出：data/wiki/quality.train 和 data/wiki/quality.valid
uv run python cs336_data/quality_classifier/06-split_train_valid.py

# Step 7：训练 fastText 分类器
#         输出：out/models/quality.bin
uv run python cs336_data/quality_classifier/07-train.py
```

#### 训练参数

```python
fasttext.train_supervised(
    input="data/wiki/quality.train",
    epoch=30,
    lr=0.2,       # 经实验，lr=0.2 比默认值效果更好
)
# 最终验证集准确率约 0.81
```

> ⚠️ **注意**：Step 3 依赖 NSFW 和毒性分类器（需要先配置好 `classifiers/` 目录下的两个 dolma 模型）。如果跳过 NSFW/毒性过滤，可以直接用语言过滤 + Gopher 过滤来筛选正例。

### 4. Common Crawl 原始数据（Leaderboard Pipeline 需要）

如果要运行完整的 Leaderboard 数据过滤 Pipeline，需要获取 Common Crawl WET 文件：
- 格式：`CC-MAIN-*.warc.wet.gz`
- 默认路径：`/data/CC/`（可通过 `--data-dir` 参数指定）
- 官方地址：[https://commoncrawl.org/](https://commoncrawl.org/)
- 文件列表：`https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz`

推荐和 WARC 一样，先下载 `wet.paths.gz`，再从清单中选择需要的 WET 文件；不要依赖某个写死的 segment 路径长期有效。

> ⚠️ **注意格式区别**：主 Pipeline（Step 1–5）使用 **WET 格式**（已提取纯文本）；质量分类器负例准备需要 **WARC 格式**（含原始 HTML）。两者均可从 commoncrawl.org 下载，路径格式不同。

#### 存储说明

本项目原始运行环境为**斯坦福 HPC 集群**，CC 原始数据放在集群共享存储 `/data/CC/` 下（课程组统一预置，学生只读访问）。个人复现时需自行下载。

各阶段数据量（基于 5,000 个 WET 文件）：

| 阶段 | 大小估算 |
|---|---|
| 原始 CC WET 文件（5,000个，压缩） | ~1 TB |
| Step 1 英文过滤后（~14%） | ~140 GB |
| Step 2–3 启发式+去重后（~4%） | ~40 GB |
| Step 4 分类器过滤后 | ~15 GB |
| **最终 tokenized `.bin`（固定产物）** | **~12.8 GB** |

> 💡 如果存储有限，可以流式处理（边过滤边写出），每次只保留当前步的输出，处理完后删除上一步的中间产物。

### 5. Paloma 验证集（Leaderboard 分类器正例 + 模型评估）

Leaderboard 分类器和最终模型评估都依赖 **Paloma benchmark** 的 `c4_100_domains` 子集。

- **HuggingFace**：[huggingface.co/datasets/allenai/paloma](https://huggingface.co/datasets/allenai/paloma)
- **GitHub**：[github.com/allenai/paloma](https://github.com/allenai/paloma)
- **访问限制**：数据集设有访问限制，需登录 HuggingFace 并同意 AI2 ImpACT License

```bash
# 由于缺乏集群，后续排行榜评估和模型验证均强依赖该文件的 tokenized 格式，
# 请安装额外依赖并运行以下 Python 脚本自动构建成所需的 .bin 文件。
# `transformers` 已经会随 `cs336-basics` 安装；这里只需额外补 `datasets`：
uv run pip install datasets
# 或：uv add datasets
```

```python
# build_paloma.py
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

print("Downloading Paloma validation dataset...")
try:
    ds = load_dataset('allenai/paloma', 'c4_100_domains', split='validation')
except ValueError:
    # 当前 Hugging Face 上该子集公开的 split 名称通常是 `val` / `test`
    ds = load_dataset('allenai/paloma', 'c4_100_domains', split='val')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

all_ids = []
for item in tqdm(ds, desc="Tokenizing Paloma"):
    # 按照代码规范，文本必须带有结束符 EOS (50256)
    tokens = tokenizer.encode(item['text']) + [tokenizer.eos_token_id]
    all_ids.extend(tokens)

out_dir = "data/paloma"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/tokenized_paloma_c4_100_domains_validation.bin"
np.array(all_ids, dtype=np.uint16).tofile(out_path)
print(f"Success! Saved {len(all_ids)} tokens to {out_path}")
```

再建立项目内兼容路径：

```bash
mkdir -p data/leaderboard
ln -sf ../paloma/tokenized_paloma_c4_100_domains_validation.bin \
    data/leaderboard/tokenized_paloma_c4_100_domains_validation.bin
```

如果你沿用 `cs336-basics/configs/experiment/*.yaml` 里的默认 `valid_bin: /data/paloma/...` 配置，还需要二选一：

1. 额外创建 `/data/paloma/tokenized_paloma_c4_100_domains_validation.bin` 的软链
2. 直接把这些配置文件里的 `valid_bin` 改成项目内路径 `data/paloma/tokenized_paloma_c4_100_domains_validation.bin`

这样即可产出模型验证时的 `valid_bin` 数据，并兼容本仓库中两套读取路径。

### 6. 质量分类器 B——Leaderboard 实际使用（`cs336_data/leaderboard/classifier/`）

⚠️ **这是 Pipeline Step 4 实际调用的分类器**，与 § 3 的测试用分类器完全独立。

两者的核心区别：

| | 测试用（§ 3） | Leaderboard 实际用（§ 6） |
|---|---|---|
| 正例来源 | 维基百科页面（爬取） | **Paloma `c4_100_domains` 验证集** |
| 负例来源 | CC WARC 随机采样 | Step 3 去重后的 CC 文本 |
| 模型路径 | `out/models/quality.bin` | `data/leaderboard/classifier/quality.bin`（个人服务器推荐） |
| 目的 | 通过单元测试 | 最终数据过滤，最小化 Paloma 验证损失 |

#### 完整训练流程（5步）

**所需原始数据**：Paloma 验证集（§ 5）+ Step 3 去重后的 CC 文本

> ⚠️ 如果你只下载了 2–3 个 WET 文件做本地小样本实验，`03-select_negatives.py` 默认的
> `28,000` 训练负例和 `500` 验证负例通常过大。此时请显式调小：
> ```bash
> uv run python cs336_data/leaderboard/classifier/03-select_negatives.py \
>     --data-dir data/03-deduped \
>     --num-train-examples 1000 \
>     --num-valid-examples 100
> ```

所有脚本默认输出到 `data/leaderboard/classifier/`，无需修改代码：

```bash
# Step 1：将 Paloma 验证集（tokenized .bin）解码为文本
#   输入：data/leaderboard/tokenized_paloma_c4_100_domains_validation.bin
#   输出：data/leaderboard/classifier/paloma_c4_100_domains_validation_text.txt
uv run python cs336_data/leaderboard/classifier/01-val_to_text.py

# Step 2：从 Paloma 文本中采样正例（28,000 训练 + 500 验证，不足时重复采样）
uv run python cs336_data/leaderboard/classifier/02-sample_positives.py

# Step 3：从 Step 3 去重后的 CC 文本中采样负例
#   --data-dir 指向你自己的 Step 3 输出目录。
#   如果你沿用下面“小规模实验”的命令，则这里应传 data/03-deduped；
#   如果你自己把 Step 3 输出放在 data/leaderboard/03-deduped，也同样可以。
uv run python cs336_data/leaderboard/classifier/03-select_negatives.py \
    --data-dir data/03-deduped

# Step 4：合并正负例，生成 quality.train 和 quality.valid
uv run python cs336_data/leaderboard/classifier/04-merge_samples.py

# Step 5：训练 fastText 分类器
#   输出：data/leaderboard/classifier/quality.bin（供 04-c4_100_classify.py 自动加载）
uv run python cs336_data/leaderboard/classifier/05-train.py
```

> 所有脚本均支持 `--classifier-dir` 参数自定义输出目录，默认为 `data/leaderboard/classifier/`。`04-c4_100_classify.py` 会自动在该路径找到模型，无需手动修改代码。

---

## 运行单元测试

```bash
# 运行所有测试（需要已配置好模型文件和 NLTK 数据）
uv run pytest tests/ -v

# 运行指定测试文件
uv run pytest tests/test_pii.py -v
uv run pytest tests/test_quality.py -v
uv run pytest tests/test_deduplication.py -v
```

> 首次运行可能需要下载 NLTK 数据：
> ```python
> import nltk; nltk.download('punkt_tab')
> ```

---

## 各功能模块使用方式

### HTML 文本提取

```python
from cs336_data.extract_text import extract_text_from_html_bytes

with open("page.html", "rb") as f:
    html_bytes = f.read()

text = extract_text_from_html_bytes(html_bytes)
print(text)
```

### 语种识别

```python
from cs336_data.language_identification import identify_language

lang, score = identify_language("Hello, this is English text.")
print(lang, score)  # en  0.9999...
```

### PII 脱敏

```python
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips

text = "Contact us at hello@example.com or call 123-456-7890."
masked, count = mask_emails(text)
print(masked)  # Contact us at |||EMAIL_ADDRESS||| or call 123-456-7890.

masked, count = mask_phone_numbers(text)
print(masked)  # Contact us at hello@example.com or call |||PHONE_NUMBER|||.
```

### 有害内容分类

```python
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

label, score = classify_nsfw("some text here")
print(label, score)  # "nsfw" or "non-nsfw", float

label, score = classify_toxic_speech("some text here")
print(label, score)  # "toxic" or "non-toxic", float
```

### Gopher 质量过滤

```python
from cs336_data.gopher_quality_filters import gopher_quality_filter

text = "This is a long enough piece of text..." * 100
is_good = gopher_quality_filter(text)
print(is_good)  # True or False
```

过滤规则：
- 词数在 50 ~ 100,000 之间
- 平均词长在 3 ~ 10 字符之间
- 以省略号结尾的行不超过 30%
- 含字母字符的词占比 ≥ 80%

### 质量分类器

```python
from cs336_data.quality_classifier.quality_classifier import classify_quality

label, score = classify_quality("Some web page text...")
print(label, score)  # "wiki" (高质量) or "cc" (低质量), float
```

### 精确去重

```python
from cs336_data.exact_deduplication import exact_line_dedupe
from pathlib import Path

input_files = list(Path("./data").glob("*.txt"))
exact_line_dedupe(input_files, output_directory=Path("./deduped"))
```

### MinHash 模糊去重

```python
from cs336_data.minhash_deduplication import minhash_dedupe
from pathlib import Path

input_files = list(Path("./data").glob("*.txt"))
minhash_dedupe(
    input_files=input_files,
    num_hashes=100,
    num_bands=10,
    ngrams=5,
    jaccard_threshold=0.8,
    output_directory=Path("./deduped"),
)
```

---

## 小规模实验（本地验证 Pipeline 逻辑）

只需下载 2–3 个 WET 文件即可验证本地流程；通常每个压缩文件约 90–100 MB，总体约 180–300 MB。

### 第一步：下载少量 WET 文件

```bash
mkdir -p data/raw data/commoncrawl

# 下载 WET 文件列表
wget -O data/commoncrawl/wet.paths.gz \
    https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz
zcat data/commoncrawl/wet.paths.gz | head -3

# 下载前 2 个 WET 文件；如果你想多留一点样本，可把 head -2 改成 head -3
zcat data/commoncrawl/wet.paths.gz | head -2 | while read -r wet_path; do
    wget -P data/raw "https://data.commoncrawl.org/${wet_path}"
done
```

> 完整文件列表：`https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz`

### 第二步：依次运行各步骤

**划重点：**原作者在所有的 `01` 到 `05` 脚本中都体贴地内置了小样本本地运行参数：
- `--max-files N`：只处理输入目录内的前 N 个文件，极大缩小测试时间。
- `--single`：禁用集群，直接在本地单线程跑完清洗流程，并打印 tqdm 进度。
- `--mp`：也是本地执行，但会调用机器的所有 CPU 核心（通过 ProcessPool）多核加速跑。

测试小段核心逻辑时，建议把 `--max-files` 显式设小：

```bash
# Step 1: 语种过滤（保留英文，p(English) > 0.85）
uv run python cs336_data/leaderboard/01-language.py \
    --data-dir data/raw \
    --out-dir  data/01-english \
    --single \
    --max-files 2

# Step 2: 启发式过滤（C4 + Gopher 规则）
uv run python cs336_data/leaderboard/02-heuristics.py \
    --data-dir data/01-english \
    --out-dir  data/02-heuristics \
    --single \
    --max-files 2

# Step 3: 精确行去重
uv run python cs336_data/leaderboard/03-exact_dedupe.py \
    --data-dir data/02-heuristics \
    --outdir   data/03-deduped

# Step 4: 质量分类器过滤（需要先训练好 data/leaderboard/classifier/quality.bin）
uv run python cs336_data/leaderboard/04-c4_100_classify.py \
    --data-dir data/03-deduped \
    --out-dir  data/04-classified \
    --single \
    --max-files 2

# Step 5: Tokenize
uv run python cs336_data/leaderboard/05-tokenize.py \
    --input-dir  data/04-classified \
    --output-path data/tokens.bin
```

### 前置依赖说明

| 步骤 | 需要的模型文件 |
|---|---|
| Step 1（语种过滤） | `data/classifiers/lid.176.bin` |
| Step 2（启发式过滤） | 无需模型文件，只用规则过滤；但依赖 `transformers`（已含在 `uv sync`） |
| Step 3（精确去重） | 无需模型文件 |
| Step 4（质量分类） | Leaderboard 分类器 `data/leaderboard/classifier/quality.bin`（需先完成 § 6 训练；脚本会自动查找，无需改代码） |
| Step 1–3 | 只需 `lid.176.bin`，可独立验证前三步 |

---

## 完整数据过滤 Pipeline（集群规模）

脚本均在 `cs336_data/leaderboard/` 目录下，按编号顺序执行。  
默认模式（不加 `--single` / `--mp`）为 **SLURM 集群并行**，需要配置 submitit 环境。

```bash
# Step 1: 语种过滤
uv run python cs336_data/leaderboard/01-language.py \
    --data-dir /data/CC \
    --out-dir  /data/output/01-english
    # 加 --single 单进程；加 --mp 本地多进程

# Step 2: 启发式过滤（C4 + Gopher 规则）
uv run python cs336_data/leaderboard/02-heuristics.py \
    --data-dir /data/output/01-english \
    --out-dir  /data/output/02-heuristics

# Step 3: 精确行去重
uv run python cs336_data/leaderboard/03-exact_dedupe.py \
    --data-dir /data/output/02-heuristics \
    --outdir   /data/output/03-deduped

# Step 4: 质量分类器过滤
uv run python cs336_data/leaderboard/04-c4_100_classify.py \
    --data-dir /data/output/03-deduped \
    --out-dir  /data/output/04-classified

# Step 5: Tokenize
uv run python cs336_data/leaderboard/05-tokenize.py \
    --input-dir  /data/output/04-classified \
    --output-path /data/output/tokens.bin
```

---

## 模型训练

训练使用 `cs336-basics` 提供的训练代码，如果是集群环境，通过 SLURM 提交：

```bash
sbatch cs336_data/leaderboard/train.sh
```

如果是本地单机环境（无需 sbatch）：

```bash
# 请根据你本地实际的 GPU 数量修改 --nproc_per_node
uv run torchrun --standalone --nproc_per_node=2 \
    scripts/train.py \
    --config-name=experiment/bucketed.yaml
```

---

## 提交

```bash
bash test_and_make_submission.sh
```

该脚本会：
1. 安装依赖
2. 运行所有单元测试
3. 将代码打包为 `.zip` 提交文件
