"""
Experiment 1: baseline LLM learner simulation and Table 1 comparison.

Default mode scores the released edit/uptake labels in experiment_1.csv.
Pass --simulate to regenerate labels with an API key (config.py required).
Prompt templates follow the learner-simulation setup described in Methods
and Supplementary Information, Section 3.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, RESULT_DIR
from utils import UPTAKE_LABELS, cramers_v, normalize_uptake

MODELS = {
    "GPT-4o": ("4o-is_essay_edited_new_new", "4o-uptake_classification"),
    "GPT-4o-mini": ("4o-mini-is_essay_edited_new_new", "4o-mini-uptake_classification"),
    "Llama-3.1-70B": ("llama-is_essay_edited_new_new", "llama-uptake_classification"),
    "Claude-3.5-Sonnet": ("claude-is_essay_edited_new_new", "claude-uptake_classification"),
}


def create_student_response_prompt(student_essay: str, student_utterance: str, teacher_feedback: str) -> str:
    return (
        "You are a student focusing on English writing. Analyze your written essay "
        "carefully, explain your thought process (1024 tokens or less), and try to "
        "apply the concepts you've learned to revise the essay. If you're unsure, "
        "express your uncertainty and explain your reasoning.\n\n"
        f"Your written Essay: [{student_essay}]\n\n"
        f"You: [{student_utterance}]\n\n"
        f"Feedback provider: [{teacher_feedback}]\n\n"
        "Your next response here:"
    )


def create_essay_revision_prompt(
    student_utterance: str,
    teacher_feedback: str,
    student_next_response: str,
    previous_essay: str,
) -> str:
    return (
        "Based on your current understanding after the conversation, rethink step "
        "by step and then revise the essay.\n\n"
        f"You: [{student_utterance}]\n\n"
        f"Feedback provider: [{teacher_feedback}]\n\n"
        f"You: [{student_next_response}]\n\n"
        f"Your previous essay: [{previous_essay}]\n\n"
        "Revised essay:"
    )


def create_diff_prompt(essay_before: str, essay_after: str) -> str:
    return f"""You are a text comparison tool. Find added, removed, or modified sentences.

For modified sentences:
- "type": "modified"
- "text": "<original sentence> -> <updated sentence>"

For added sentences:
- "type": "added"
- "text": "<added sentence>"

For removed sentences:
- "type": "removed"
- "text": "<removed sentence>"

Return JSON only, with no extra explanation.

essay_before = {essay_before}
essay_after = {essay_after}

output="""


def create_uptake_classification_prompt(corrective_feedback: str, revision_record: str) -> str:
    return f"""You are classifying learner uptake after corrective feedback.

Definitions:
- SUCCESSFUL: The learner corrected the error(s) after receiving feedback, and no further repair is needed.
- NO-UPTAKE: The learner ignored the feedback entirely, did not revise based on it, and made no engagement with the suggested changes.
- UNSUCCESSFUL: The learner tried to correct the error(s) using the feedback but failed to make sufficient improvements.

Task:
Classify the learner uptake into exactly one of: SUCCESSFUL, NO-UPTAKE, UNSUCCESSFUL.

corrective_feedback: {corrective_feedback}
learner's revision record: {revision_record}

Answer with only one label."""


def percent_counts(series: pd.Series) -> dict:
    counts = series.value_counts()
    n = len(series)
    return {label: 100.0 * counts.get(label, 0) / n for label in UPTAKE_LABELS}


def analyze_released_labels() -> None:
    df = pd.read_csv(DATA_DIR / "experiment_1.csv")
    df = df[df["is_corrective_feedback"] == True].copy()
    n = len(df)
    print(f"Corrective-feedback pairs: {n}")

    human_edit = df["is_essay_edited_new"].astype(bool)
    human_uptake = df["uptake_classification"].map(normalize_uptake)
    rows = [
        {
            "source": "Real learners",
            "n": n,
            "no_edit_pct": 100.0 * (~human_edit).mean(),
            "with_edit_pct": 100.0 * human_edit.mean(),
            "successful_pct": percent_counts(human_uptake)["SUCCESSFUL"],
            "no_uptake_pct": percent_counts(human_uptake)["NO-UPTAKE"],
            "unsuccessful_pct": percent_counts(human_uptake)["UNSUCCESSFUL"],
        }
    ]

    print("\nTable 1. Essay-edit and uptake behaviour")
    print(f"{'Source':<22} {'No edit':>8} {'Edit':>8} {'Successful':>12} {'No uptake':>10} {'Unsuccessful':>13}")
    print("-" * 75)
    print(
        f"{'Real learners':<22} {rows[0]['no_edit_pct']:8.2f} {rows[0]['with_edit_pct']:8.2f} "
        f"{rows[0]['successful_pct']:12.2f} {rows[0]['no_uptake_pct']:10.2f} {rows[0]['unsuccessful_pct']:13.2f}"
    )

    llm_edit_values = []
    llm_uptake_tables = []
    for model, (edit_col, uptake_col) in MODELS.items():
        edited = df[edit_col].astype(bool)
        uptake = df[uptake_col].map(normalize_uptake)
        pct = percent_counts(uptake)
        row = {
            "source": model,
            "n": n,
            "no_edit_pct": 100.0 * (~edited).mean(),
            "with_edit_pct": 100.0 * edited.mean(),
            "successful_pct": pct["SUCCESSFUL"],
            "no_uptake_pct": pct["NO-UPTAKE"],
            "unsuccessful_pct": pct["UNSUCCESSFUL"],
        }
        rows.append(row)
        print(
            f"{model:<22} {row['no_edit_pct']:8.2f} {row['with_edit_pct']:8.2f} "
            f"{row['successful_pct']:12.2f} {row['no_uptake_pct']:10.2f} {row['unsuccessful_pct']:13.2f}"
        )
        llm_edit_values.append(edited.to_numpy())
        llm_uptake_tables.append(uptake.to_numpy())

    pd.DataFrame(rows).to_csv(RESULT_DIR / "baseline_llm_table1.csv", index=False)

    llm_edit = np.concatenate(llm_edit_values)
    edit_contingency = np.array(
        [[(~human_edit).sum(), human_edit.sum()], [(~llm_edit).sum(), llm_edit.sum()]]
    )
    chi2_edit, p_edit, dof_edit, _ = stats.chi2_contingency(edit_contingency)

    uptake_by_model = [
        pd.Series(vals).value_counts().reindex(UPTAKE_LABELS, fill_value=0) for vals in llm_uptake_tables
    ]
    llm_uptake_contingency = pd.concat(uptake_by_model, axis=1).to_numpy()
    chi2_hom, p_hom, dof_hom, _ = stats.chi2_contingency(llm_uptake_contingency)

    llm_uptake = np.concatenate(llm_uptake_tables)
    human_counts = human_uptake.value_counts().reindex(UPTAKE_LABELS, fill_value=0)
    llm_counts = pd.Series(llm_uptake).value_counts().reindex(UPTAKE_LABELS, fill_value=0)
    uptake_contingency = np.vstack([human_counts.to_numpy(), llm_counts.to_numpy()])
    chi2_up, p_up, dof_up, _ = stats.chi2_contingency(uptake_contingency)

    tests = pd.DataFrame(
        [
            {
                "test": "human_vs_llm_edit",
                "chi2": chi2_edit,
                "df": dof_edit,
                "p_value": p_edit,
                "n": edit_contingency.sum(),
                "cramers_v": cramers_v(chi2_edit, edit_contingency.sum(), *edit_contingency.shape),
            },
            {
                "test": "llm_uptake_homogeneity",
                "chi2": chi2_hom,
                "df": dof_hom,
                "p_value": p_hom,
                "n": llm_uptake_contingency.sum(),
                "cramers_v": cramers_v(chi2_hom, llm_uptake_contingency.sum(), *llm_uptake_contingency.shape),
            },
            {
                "test": "human_vs_llm_uptake",
                "chi2": chi2_up,
                "df": dof_up,
                "p_value": p_up,
                "n": uptake_contingency.sum(),
                "cramers_v": cramers_v(chi2_up, uptake_contingency.sum(), *uptake_contingency.shape),
            },
        ]
    )
    tests.to_csv(RESULT_DIR / "baseline_llm_chisquare.csv", index=False)
    print("\nChi-square tests")
    for row in tests.itertuples(index=False):
        print(f"  {row.test}: chi2({row.df}) = {row.chi2:.2f}, p = {row.p_value:.4g}, V = {row.cramers_v:.3f}")


def run_simulation(input_path: Path, output_path: Path) -> None:
    """Regenerate baseline LLM responses. Requires config.py and openai."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        import config
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Simulation requires config.py and the openai package. "
            "Copy config.example.py to config.py and install openai."
        ) from exc

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    df = pd.read_csv(input_path)
    responses = []
    essays = []
    diffs = []
    labels = []
    for _, row in df.iterrows():
        response = client.chat.completions.create(
            model=config.STUDENT_MODEL,
            messages=[{"role": "user", "content": create_student_response_prompt(row["prev_essay"], row["user"], row["chatgpt_after"])}],
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
        ).choices[0].message.content
        responses.append(response)
        essay = client.chat.completions.create(
            model=config.STUDENT_MODEL,
            messages=[{"role": "user", "content": create_essay_revision_prompt(row["user"], row["chatgpt_after"], response, row["prev_essay"])}],
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
        ).choices[0].message.content
        essays.append(essay)
        edited = str(row["prev_essay"]).strip() != str(essay).strip()
        if edited:
            diff = client.chat.completions.create(
                model=config.DIFF_CHECK_MODEL,
                messages=[{"role": "user", "content": create_diff_prompt(row["prev_essay"], essay)}],
                temperature=config.TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            ).choices[0].message.content
            label = client.chat.completions.create(
                model=config.DIFF_CHECK_MODEL,
                messages=[{"role": "user", "content": create_uptake_classification_prompt(row["chatgpt_after"], diff)}],
                temperature=config.TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            ).choices[0].message.content
        else:
            diff = "No difference"
            label = "NO-UPTAKE"
        diffs.append(diff)
        labels.append(str(label).strip())

    df["llm_next_response"] = responses
    df["llm_essay_generated"] = essays
    df["llm_sentence_diff"] = diffs
    df["llm_uptake_classification"] = labels
    df.to_csv(output_path, index=False)
    print(f"Wrote simulated outputs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 baseline LLM comparison")
    parser.add_argument("--simulate", action="store_true", help="Regenerate labels with an API key")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.simulate:
        input_path = Path(args.input) if args.input else DATA_DIR / "experiment_1.csv"
        output_path = Path(args.output) if args.output else RESULT_DIR / "baseline_llm_simulated.csv"
        run_simulation(input_path, output_path)
        return

    analyze_released_labels()
    print(f"\nWrote results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
