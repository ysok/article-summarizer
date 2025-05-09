
# 📰 News Article Summarizer

This Python script reads news articles from `.txt` files and summarizes them into concise, readable paragraphs using different choice of AI model. In this example we use [IBM Granite 3.2 8B Instruct](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct) large language model. Once all the models are downloaded, we run this AI model offline.

## ✨ What this project is a about?

News articles can be long and time-consuming to read. This tool helps you:

- Quickly digest lengthy news content.
- Focus only on the key points.
- Save time without missing the story.

## 🖥️ Requirements

To run this script **locally** (Tested on Macbook M3), you’ll need:

- Python 3.10 or newer
- [Conda](https://docs.conda.io/en/latest/) (recommended for environment isolation)
- At least **32 GB RAM** recommended
- Machine with GPU acceleration

> ✅ Tested on MacBook M3 with macOS Sonoma

## 📦 Installation (with Conda)

Follow these steps to set up your environment:

### 1. Install Miniconda or Anaconda
If not already installed, download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for macOS (Apple Silicon).

### 2. Create a Conda environment
```bash
conda create -n article-summarizer python=3.10 -y
conda activate article-summarizer
````

### 3. Install PyTorch (with MPS support)

```bash
pip install torch torchvision torchaudio
```

> Make sure PyTorch detects MPS (Metal):
>
> ```python
> import torch
> torch.backends.mps.is_available()
> ```

### 4. Install other dependencies

```bash
pip install -r requirements.txt
```

Make sure to login to huggingface, accept all agreements and create your login token.
It will prompt you to paste your Access Token. You can find or create your token here:
👉 https://huggingface.co/settings/tokens
```
huggingface-cli login
```


## ✏️ How to Use

1. Drop all your `.txt` news articles into `sample/input/`.
2. Run the script:

```bash
python summarizer.py
```

3. Summarized results will be printed to the console and also saved to `sample/output/` with filenames like `article-out.txt`.

## 🧠 Model Details

* Model: `ibm-granite/granite-3.2-8b-instruct`
* Type: Causal Language Model
* Hosted on: [Hugging Face](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct)
* Runs locally on CPU or MPS (Apple Silicon GPU)

## ⚙️ Example Output

**Input (sample/input/news1.txt):**

> "The central bank announced another interest rate hike amid fears of rising inflation..."

**Output (sample/output/news1-out.txt):**

> \----> Summary (248 chars, 1.32s):
> The central bank increased interest rates to combat inflation, signaling concern about the economy. Analysts expect further hikes if inflation remains persistent.

---

## 🚀 Future Ideas

* Read realtime news API and summarize them
* Send out summrized articles to receipients
* Support batch summarization in parallel
* Add web interface for uploads and summary

## 📖 License

This project is open-sourced under the MIT License.

## Maintainers
* https://github.com/ysok

