"""
LLM-peer essay generation protocol and range-validation analysis (Fig. 2).

Default mode compares released learner and peer essays in experiment_2_essay.csv.
Pass --generate to call the staged prompting protocol. Exact JSON schemas used
in the classroom deployment are in Supplementary Information, Section 4.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, FIGURE_DIR, RESULT_DIR
from utils import (
    flesch_reading_ease,
    jensen_shannon_divergence,
    kolmogorov_smirnov_test,
    parse_kc_list,
    range_overlap_proportion,
    tost_equivalence_test,
    type_token_ratio,
    wasserstein_distance_normalized,
    word_count,
)

FEATURES = [
    ("kc_count", "Number of incorrect KC"),
    ("word_count", "Word count"),
    ("ttr", "Type-token ratio"),
    ("fres", "Flesch Reading Ease Score"),
]

UPTAKE_WEIGHTS = {
    "low-uptake": {"NO-UPTAKE": 1.0, "SUCCESSFUL": 0.0, "UNSUCCESSFUL": 0.0},
    "high-uptake": {"NO-UPTAKE": 0.143, "SUCCESSFUL": 0.714, "UNSUCCESSFUL": 0.143},
    "mixed-uptake": {"NO-UPTAKE": 0.444, "SUCCESSFUL": 0.222, "UNSUCCESSFUL": 0.333},
}


def create_error_extraction_prompt(student_essay: str) -> str:
    return f"""Extract incorrect knowledge components from the learner essay.
Use the coding scheme spanning content, organization, and language.
Return a JSON list. Each item must include description, kc_id, evidence, priority, and category.

Essay:
{student_essay}
"""


def create_peer_essay_prompt(student_essay: str, knowledge_components: str, topic: str) -> str:
    return f"""Write a new essay on the topic below that matches the learner essay in
length, style, and linguistic sophistication. Then insert the listed erroneous
knowledge components so the peer essay mirrors the learner error profile without
copying the original text.

Topic: {topic}

Learner essay:
{student_essay}

Erroneous knowledge components:
{knowledge_components}
"""


def create_uptake_reply_prompt(peer_essay: str, learner_feedback: str, uptake_type: str) -> str:
    return f"""You are an LLM peer receiving corrective feedback on your essay.
Respond according to the sampled uptake type: {uptake_type}.

- SUCCESSFUL: revise the identified error as requested.
- UNSUCCESSFUL: attempt a revision that still falls short.
- NO-UPTAKE: do not apply the suggested change; ask for clarification or resist.

Peer essay:
{peer_essay}

Learner feedback:
{learner_feedback}
"""


def sample_uptake_type(profile: str, rng: np.random.Generator) -> str:
    weights = UPTAKE_WEIGHTS[profile]
    labels = list(weights)
    return rng.choice(labels, p=[weights[label] for label in labels])


def essay_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "uid": row.get("uid"),
                "group_type": row.get("group_type"),
                "human_kc_count": len(parse_kc_list(row["studentJson"])),
                "llm_kc_count": len(parse_kc_list(row["peerJson"])),
                "human_word_count": word_count(row["studentEssay"]),
                "llm_word_count": word_count(row["peerEssay"]),
                "human_ttr": type_token_ratio(row["studentEssay"]),
                "llm_ttr": type_token_ratio(row["peerEssay"]),
                "human_fres": flesch_reading_ease(row["studentEssay"]),
                "llm_fres": flesch_reading_ease(row["peerEssay"]),
            }
        )
    return pd.DataFrame(records)


def validate_peer_essays() -> None:
    df = pd.read_csv(DATA_DIR / "experiment_2_essay.csv")
    features = essay_features(df)
    features.to_csv(RESULT_DIR / "peer_essay_features.csv", index=False)
    print(f"Essay pairs: {len(features)}")

    rows = []
    for key, label in FEATURES:
        human = features[f"human_{key}"].dropna().to_numpy()
        llm = features[f"llm_{key}"].dropna().to_numpy()
        delta = 0.5 * np.std(human, ddof=1)
        tost = tost_equivalence_test(llm, human, delta=delta)
        ks_d, ks_p = kolmogorov_smirnov_test(human, llm)
        row = {
            "feature": key,
            "label": label,
            "n_pairs": len(human),
            "human_mean": float(np.mean(human)),
            "llm_mean": float(np.mean(llm)),
            "mean_diff": tost["mean_diff"],
            "ci_lower": tost["ci_lower"],
            "ci_upper": tost["ci_upper"],
            "tost_p": tost["p_tost"],
            "ks_d": ks_d,
            "ks_p": ks_p,
            "wasserstein_sd": wasserstein_distance_normalized(human, llm),
            "js_divergence": jensen_shannon_divergence(human, llm),
            "within_human_range": range_overlap_proportion(human, llm),
        }
        rows.append(row)
        print(
            f"  {label}: diff = {row['mean_diff']:.2f}, "
            f"TOST p = {row['tost_p']:.4g}, KS D = {row['ks_d']:.2f}, "
            f"range overlap = {row['within_human_range']:.1%}"
        )

    pd.DataFrame(rows).to_csv(RESULT_DIR / "peer_essay_validation.csv", index=False)
    try:
        _plot_feature_distributions(features)
        print(f"Wrote {FIGURE_DIR / 'peer_essay_validation.pdf'}")
    except Exception as exc:
        print(f"Could not write figure: {exc}")


def _plot_feature_distributions(features: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, (key, label) in enumerate(FEATURES):
        human = features[f"human_{key}"].dropna()
        llm = features[f"llm_{key}"].dropna()
        axes[0, i].boxplot([human, llm])
        axes[0, i].set_xticklabels(["Human", "LLM peer"])
        axes[0, i].set_title(label)
        axes[0, i].set_ylabel("Value")
        axes[1, i].hist(human, bins=20, density=True, alpha=0.5, label="Human")
        axes[1, i].hist(llm, bins=20, density=True, alpha=0.5, label="LLM peer")
        axes[1, i].set_xlabel(label)
        axes[1, i].set_ylabel("Density")
        if i == 0:
            axes[1, i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "peer_essay_validation.pdf")
    plt.close(fig)


def generate_peer_essay(student_essay: str, topic: str, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    extraction = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": create_error_extraction_prompt(student_essay)}],
    ).choices[0].message.content
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": create_peer_essay_prompt(student_essay, extraction, topic)}],
    ).choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-peer essay generation and validation")
    parser.add_argument("--generate", action="store_true", help="Call the generation protocol")
    parser.add_argument("--essay", type=str, default="", help="Learner essay text for --generate")
    parser.add_argument("--topic", type=str, default="cooking at home")
    args = parser.parse_args()

    if args.generate:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            import config
        except ImportError as exc:
            raise SystemExit("Copy config.example.py to config.py before using --generate.") from exc
        if not args.essay:
            raise SystemExit("Provide --essay when using --generate.")
        print(generate_peer_essay(args.essay, args.topic, config.OPENAI_API_KEY, config.DEFAULT_MODEL))
        return

    validate_peer_essays()


if __name__ == "__main__":
    main()
