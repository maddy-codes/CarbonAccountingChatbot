import json
import os
import re
import string
import sys
import time
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import util

from src.models.generator import CarbonChatbot  # Your chatbot class


def calculate_semantic_similarity(prediction, ground_truth, model):
    # Encoding both of the sentences into the vector
    emb1 = model.encode(prediction)
    emb2 = model.encode(ground_truth)

    # Calculating cosine similarity
    cos_sim = util.cos_sim(emb1, emb2)

    return cos_sim.item()


def normalize_text(s):
    """Lowercases, removes punctuation, and strips whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1(prediction, ground_truth):
    """Calculates token-level overlap between predicted and reference answers."""
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_em(prediction, ground_truth):
    """Checks for exact string match after normalization."""
    return normalize_text(prediction) == normalize_text(ground_truth)


def run_evaluation():
    # Initialize your bot artifact
    bot = CarbonChatbot()

    with open("tests/test_set.json", "r") as f:
        test_set = json.load(f)

    metrics = {
        "f1_scores": [],
        "em_scores": [],
        "latency_scores": [],
        "citation_correct": 0,
        "semantic_scores": [],
    }

    print(f"🚀 Starting evaluation on {len(test_set)} test cases...")

    for i, item in enumerate(test_set):
        start_time = time.time()

        # Query the chatbot
        response = bot.ask(item["question"])
        latency = time.time() - start_time

        # Calculate scores
        f1 = calculate_f1(response["answer"], item["expected_answer"])
        em = calculate_em(response["answer"], item["expected_answer"])
        semantic_score = calculate_semantic_similarity(
            response["answer"], item["expected_answer"], bot.embedder
        )

        # Citation Check: Was the expected PDF mentioned?
        citation_found = any(
            item["expected_source"].lower() in src.lower()
            for src in response["citations"]
        )

        metrics["semantic_scores"].append(semantic_score)
        metrics["f1_scores"].append(f1)
        metrics["em_scores"].append(em)
        metrics["latency_scores"].append(latency)
        if citation_found:
            metrics["citation_correct"] += 1

        print(
            f"[{i+1}/{len(test_set)}] Q: {item['question'][:40]}... | SC: {semantic_score:.2f} | F1: {f1:.2f} | Latency: {latency:.2f}s"
        )

    # Final Report
    avg_f1 = sum(metrics["f1_scores"]) / len(test_set)
    avg_em = sum(metrics["em_scores"]) / len(test_set)
    avg_latency = sum(metrics["latency_scores"]) / len(test_set)
    avg_sc = sum(metrics["semantic_scores"]) / len(test_set)
    cit_accuracy = metrics["citation_correct"] / len(test_set)

    print("\n" + "=" * 30)
    print("📈 FINAL EVALUATION REPORT")
    print("=" * 30)
    print(f"Average Semantic Similarity Score:   {avg_sc:.2%} (Target: ≥0.8)")
    print(f"Average F1 Score:   {avg_f1:.2%} (Target: ≥85%)")
    print(f"Exact Match (EM):   {avg_em:.2%} (Target: ≥70%)")
    print(f"Citation Accuracy:  {cit_accuracy:.2%} (Target: 100%)")
    print(f"Average Latency:    {avg_latency:.2f}s (Target: <2s)")
    print("=" * 30)


if __name__ == "__main__":
    run_evaluation()
