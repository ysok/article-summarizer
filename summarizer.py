import os
import time
import torch
import warnings
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


class ArticleSummarizer:
    """
    Summarizes news articles using the IBM Granite model.
    """

    MODEL_ID = "ibm-granite/granite-3.2-8b-instruct"
    INPUT_DIR = "sample/input"
    OUTPUT_DIR = "sample/output"
    MAX_INPUT_TOKENS = 1500
    MAX_SUMMARY_TOKENS = 100

    def __init__(self):
        warnings.filterwarnings("ignore")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float32  # required for MPS stability
        ).to(self.device).eval()

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def summarize(self, article_text: str, max_new_tokens: int = MAX_SUMMARY_TOKENS) -> str:
        """
        Generates a summary for the given article text.
        """
        prompt = f"Summarize the following article:\n\n{article_text.strip()}\n\nSummary:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.MAX_INPUT_TOKENS)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True
            )

        summary = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return summary

    def load_articles(self) -> Dict[str, str]:
        """
        Reads all .txt files from the input directory.
        """
        if not os.path.isdir(self.INPUT_DIR):
            raise FileNotFoundError(f"Input directory not found: {self.INPUT_DIR}")

        articles = {}
        for filename in os.listdir(self.INPUT_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(self.INPUT_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        articles[filename] = file.read()
                except Exception as e:
                    print(f"[ERROR] Failed to read {filename}: {e}")
        return articles

    def process(self):
        """
        Processes all input articles and generates summaries.
        """
        articles = self.load_articles()
        total_start = time.time()

        for filename, content in articles.items():
            if not content:
                print(f"[WARNING] Skipping empty article: {filename}")
                continue

            print(f"\n[INFO] ============> Processing: {filename} ({len(content)} characters)")
            start = time.time()
            summary = self.summarize(content)
            duration = time.time() - start

            output_text = (
                f"----> Summary ({len(summary)} chars, {duration:.2f}s):\n{summary}"
            )
            print(output_text)

            output_path = os.path.join(self.OUTPUT_DIR, filename.replace(".txt", "-out.txt"))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)

        print(f"\n[INFO] All done in {time.time() - total_start:.2f} seconds.")


if __name__ == "__main__":
    summarizer = ArticleSummarizer()
    summarizer.process()
