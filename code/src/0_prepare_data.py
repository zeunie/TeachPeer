"""Validate released data files and report the sample sizes used in the paper."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, RESULT_DIR

FILES = {
    "experiment_1.csv": {
        "required": [
            "sample_id",
            "is_corrective_feedback",
            "is_essay_edited_new",
            "uptake_classification",
        ],
        "expected_rows": 549,
    },
    "experiment_2_chat.csv": {
        "required": ["uid", "role", "session", "group_type", "strategy", "icap"],
        "expected_rows": 7122,
    },
    "experiment_2_essay.csv": {
        "required": ["uid", "studentEssay", "studentJson", "peerEssay", "peerJson", "group_type"],
        "expected_rows": 171,
    },
    "cognitive_load_data.csv": {
        "required": [
            "participant_id",
            "session",
            "group_type",
            "intrinsic_load",
            "extraneous_load",
            "germane_load",
            "essay_improvement",
        ],
        "expected_rows": 171,
    },
}


def validate_file(name: str, spec: dict) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    missing = [col for col in spec["required"] if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return {"file": name, "rows": len(df), "expected_rows": spec["expected_rows"], "columns": len(df.columns)}


def main() -> None:
    print(f"Data directory: {DATA_DIR}")
    rows = []
    for name, spec in FILES.items():
        info = validate_file(name, spec)
        rows.append(info)
        status = "ok" if info["rows"] == info["expected_rows"] else "check"
        print(f"  {name}: {info['rows']} rows (expected {info['expected_rows']}) [{status}]")

    exp1 = pd.read_csv(DATA_DIR / "experiment_1.csv")
    exp1["student_id"] = exp1["sample_id"].astype(str).str.split("-").str[0]
    chat = pd.read_csv(DATA_DIR / "experiment_2_chat.csv")
    user = chat[chat["role"] == "user"]
    load = pd.read_csv(DATA_DIR / "cognitive_load_data.csv")

    summary = {
        "experiment_1_pairs": len(exp1),
        "experiment_1_learners": exp1["student_id"].nunique(),
        "experiment_2_turns": len(chat),
        "experiment_2_sessions": chat.groupby(["uid", "session"]).ngroups,
        "experiment_2_learners": chat["uid"].nunique(),
        "experiment_2_user_utterances": len(user),
        "experiment_2_icap_others": int((user["icap"] == "Others").sum()),
        "cognitive_load_participants": load["participant_id"].nunique(),
    }
    print("\nSample sizes")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    pd.DataFrame(rows).to_csv(RESULT_DIR / "data_validation.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULT_DIR / "data_sample_sizes.csv", index=False)
    print(f"\nWrote validation reports to {RESULT_DIR}")


if __name__ == "__main__":
    main()
