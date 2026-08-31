"""Render a needle-in-a-haystack heatmap from a results.json.

Score = word overlap between the model response and the expected answer.
Reads the flat {"length-depth": response} format written by run_needle.py.
"""

import json
import os
import glob
import argparse
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

prop_16 = font_manager.FontProperties(size=45)
prop_13 = font_manager.FontProperties(size=38)


def round_to_nearest_k(x, k=8000):
    return int(round(x / k)) * k


def tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', '', text.lower()).split()


NAME_MAP = {
    'flashattn': 'FlashAttention',
    'flashattention': 'FlashAttention',
    'xattention': 'XAttention',
    'xattn': 'XAttention',
    'sparse_reuse': 'SparseReuse',
    'sparse-reuse': 'SparseReuse',
    'pbs': 'PBS',
    'meanpooling': 'MeanPooling',
    'meanpool': 'MeanPooling',
    'minference': 'MInference',
    'flexprefill': 'FlexPrefill',
    'dense': 'Dense',
}


def main(args):
    path = args.eval_path
    save_dir = args.save_dir

    dir_name = path.rstrip('/').split('/')[-1]
    model_tag = args.model.rstrip('/').split('/')[-1]
    if dir_name.startswith(model_tag + '_'):
        model = model_tag
        method = dir_name[len(model_tag) + 1:].lower()
    else:
        model = model_tag
        method = dir_name.lower()

    save_dir = f"{save_dir}/{model}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dir_name}.{'pdf' if args.pdf else 'png'}"

    round_k = 8192
    if "mistral" in model.lower():
        round_k = 4096

    expected_answer = tokenize(args.expected_answer)

    data = []
    flat_json = glob.glob(f"{path}/*.json")
    for candidate in flat_json:
        with open(candidate, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                continue
        if json_data and all(re.match(r'^\d+-\d+$', k) for k in list(json_data.keys())[:5]):
            for key, response in json_data.items():
                length_str, depth_str = key.split('-')
                resp_words = tokenize(response) if response else []
                score = len(set(resp_words).intersection(set(expected_answer))) / len(set(expected_answer)) * 100
                data.append({
                    "Document Depth": int(depth_str),
                    "Context Length": round_to_nearest_k(int(length_str), round_k),
                    "Score": score,
                })
            break

    if not data:
        raise SystemExit(f"No flat results.json found under {path}")

    df = pd.DataFrame(data)
    locations = sorted(df["Context Length"].unique())
    pivot_table = pd.pivot_table(
        df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    plt.figure(figsize=(21, 15))
    ax = sns.heatmap(
        pivot_table, fmt="g", cmap=cmap,
        cbar_kws={'shrink': 0.4, 'location': 'right', 'pad': 0.02},
        linewidths=0.5, linecolor='grey', linestyle='--', vmin=0, vmax=100,
    )
    ax.set_aspect('auto')

    if "mistral" in model.lower():
        tick_labels = [f"{x/1024:.1f}k" if x % 1024 != 0 else f"{int(x/1024)}k" for x in locations]
    else:
        tick_labels = [f"{x//1024}k" for x in locations]
    ax.set_xticklabels(tick_labels, rotation=45, fontproperties=prop_13)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop_13)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=38)

    method_title = NAME_MAP.get(method, method.capitalize())
    method_title_latex = method_title.replace(" ", r"\ ")
    title_str = rf"$\bf{{{method_title_latex}}}$ Average Score : {df['Score'].mean():.2f}"

    plt.title(title_str, fontproperties=prop_16)
    plt.ylabel('Depth Percent', fontproperties=prop_16)
    plt.xlabel("Context Length", fontproperties=prop_16)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}  (avg score {df['Score'].mean():.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--save_dir", default="results/niah/vis", type=str)
    parser.add_argument("--eval_path", default="", type=str)
    parser.add_argument("--pdf", action='store_true')
    parser.add_argument("--expected_answer", type=str,
                        default="eat a sandwich and sit in Dolores Park on a sunny day.")
    args = parser.parse_args()
    main(args)
