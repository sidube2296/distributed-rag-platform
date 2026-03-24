import json
import os

def extract_arxiv_data(json_path, save_path="data/arxiv", limit=1000):

    os.makedirs(save_path, exist_ok=True)

    count = 0

    with open(json_path, "r") as f:
        for line in f:
            paper = json.loads(line)

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            text = f"{title}\n\n{abstract}"

            with open(f"{save_path}/doc_{count}.txt", "w") as out:
                out.write(text)

            count += 1

            if count >= limit:
                break

    print(f"Saved {count} papers to {save_path}")

if __name__ == "__main__":
    extract_arxiv_data("data/arxiv.json")
