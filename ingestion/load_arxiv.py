from datasets import load_dataset
import os

def load_arxiv_data(save_path="data/arxiv"):

    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset("scientific_papers", "arxiv", split="train[:50]")

    for i, item in enumerate(dataset):
        text = item["article"]

        with open(f"{save_path}/doc_{i}.txt", "w") as f:
            f.write(text)

    print(f"Saved {len(dataset)} documents to {save_path}")
