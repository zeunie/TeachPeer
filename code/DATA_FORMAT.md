# Data Format Specification

This document describes the expected format for input data files.

## Experiment 1 Data Format

**File**: `data/0821_simple_df.csv` or `data/experiment_data_cf.csv`

### Required Columns

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| `sample_id` | string | Unique identifier for each interaction | "20185052-SW-1-1-02" |
| `student_id` | int/string | Student identifier | 20185052 |
| `course` | string | Course code | "SW" or "AW" |
| `week` | int | Week number (1-7) | 1 |
| `session` | int | Session number within week | 1 |
| `idx` | int | Turn index within session | 2 |
| `user` | string | Student's utterance | "Ok. One example sentence might be..." |
| `chatgpt_after` | string | LLM teacher's feedback | "Great example! In this sentence..." |
| `prev_essay` | string | Previous version of student's essay | "In one sense, the ambiguity..." |
| `essay` | string | Current version of student's essay | "In one sense, the ambiguity..." |
| `is_corrective_feedback` | bool | Whether feedback is corrective | True/False |
| `uptake_classification` | string | Uptake category (if pre-labeled) | "SUCCESSFUL", "NO-UPTAKE", "UNSUCCESSFUL" |

### Optional Columns

| Column Name | Type | Description |
|------------|------|-------------|
| `rating` | int | Quality rating (1-5) |
| `intent_final` | string | Intent classification |
| `is_essay_edited_new` | bool | Whether essay was edited |

### Example Row

```csv
sample_id,student_id,course,week,session,idx,user,chatgpt_after,prev_essay,essay,is_corrective_feedback,uptake_classification
20185052-SW-1-1-02,20185052,SW,1,1,2,"Ok. One example sentence might be: 'Because I skipped breakfast, I felt hungry during class.'","Great example! In this sentence, 'because I skipped breakfast' is a dependent clause...","In one sense, the ambiguity of Intel syntax (dst, src) causes problems...","In one sense, the ambiguity of Intel syntax (dst, src) causes problems...",True,NO-UPTAKE
```

### Data Validation

The script `0_prepare_data.py` will check:
- ✓ All required columns present
- ✓ No completely empty essays
- ✓ Student IDs are consistent
- ✓ Temporal ordering is valid (week → session → idx)

---

## Experiment 2 Data Format

**File**: `data/experiment2_raw.csv` or `data/experiment2_interactions.csv`

### Required Columns

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| `student_id` | int/string | Student identifier | 20185052 |
| `session` | int | Session number (1-3) | 1 |
| `uptake_profile` | string | Assigned peer profile | "low-uptake", "high-uptake", "mixed-uptake" |

### Optional Columns (for strategy analysis)

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| `turn_id` | int | Turn number in interaction | 5 |
| `learner_utterance` | string | Learner's feedback to peer | "You should change 'because' to 'since' here..." |
| `instructional_strategy` | string | Coded strategy | "Reformulate", "Amplification", "Accept", etc. |

### Optional Columns (for cognitive load analysis)

| Column Name | Type | Description | Range |
|------------|------|-------------|-------|
| `intrinsic_load` | float | Intrinsic cognitive load rating | 1-10 |
| `extraneous_load` | float | Extraneous cognitive load rating | 1-10 |
| `germane_load` | float | Germane cognitive load rating | 1-10 |

### Instructional Strategy Codes

Based on Nassaji & Wells (2000):

1. **Reformulate**: Direct rephrasing or correction
2. **Amplification**: Elaborating or extending an idea
3. **Reject**: Explicit disagreement or challenge
4. **Accept**: Agreement or acknowledgment
5. **Exemplification**: Providing examples
6. **Connection**: Linking to prior knowledge
7. **Metacognitive**: Reflection on thinking process
8. **Praise**: Encouragement (coded but excluded from main analysis)

### ICAP Mapping

Strategies are automatically mapped to ICAP levels:

- **Constructive**: Amplification, Metacognitive, Connection
- **Active**: Reformulate, Exemplification, Reject
- **Passive**: Accept, Praise

### Example Row

```csv
student_id,session,uptake_profile,turn_id,learner_utterance,instructional_strategy,intrinsic_load,extraneous_load,germane_load
20185052,1,low-uptake,5,"You should change 'because' to 'since' here because 'because' is too informal for academic writing.",Reformulate,6.5,4.2,7.1
```

---

## Data Quality Checks

### Experiment 1 (N=115 learners, 1,098 corrective feedback interactions)

Expected statistics:
- **Students**: 115 unique IDs
- **Interactions**: ~1,000-2,000 total
- **Corrective feedback**: 40-60% of all interactions
- **Uptake distribution**:
  - NO-UPTAKE: 60-80%
  - SUCCESSFUL: 10-20%
  - UNSUCCESSFUL: 10-20%

### Experiment 2 (N=57 learners, 171 sessions, 7,123 turns)

Expected statistics:
- **Students**: 57 unique IDs
- **Sessions per student**: 3 (typically)
- **Turns per session**: 20-50
- **Profile distribution**: Roughly balanced across low/high/mixed uptake
- **Cognitive load**: 1-10 scale, means typically 4-7

---

## Missing Data Handling

### Acceptable Missing Values

- `uptake_classification`: Will be computed by the pipeline
- `instructional_strategy`: Can be coded after data collection
- Cognitive load columns: Not required for all analyses

### Unacceptable Missing Values

- `student_id`: Required for all rows
- `user` and `chatgpt_after`: Required for Experiment 1
- `prev_essay` and `essay`: Required for uptake analysis
- `uptake_profile`: Required for Experiment 2

---

## Data Preparation Steps

1. **Export from source system**: Export interaction logs with timestamps
2. **Anonymize**: Remove identifying information, assign student IDs
3. **Sort**: Order by student_id → week → session → idx
4. **Validate**: Run `python src/0_prepare_data.py`
5. **Annotate** (if needed): Code instructional strategies using coding scheme
6. **Verify**: Check summary statistics match expected ranges

---

## Sample Data Generator

For testing purposes, you can generate synthetic sample data:

```python
import pandas as pd
import numpy as np

# Generate sample Experiment 1 data
n_students = 10
n_interactions_per_student = 10

data = []
for student_id in range(20180001, 20180001 + n_students):
    for idx in range(1, n_interactions_per_student + 1):
        data.append({
            'sample_id': f"{student_id}-SW-1-1-{idx:02d}",
            'student_id': student_id,
            'course': 'SW',
            'week': 1,
            'session': 1,
            'idx': idx,
            'user': f"Sample student utterance {idx}",
            'chatgpt_after': f"Sample teacher feedback {idx}",
            'prev_essay': "Sample essay text before revision...",
            'essay': "Sample essay text after revision...",
            'is_corrective_feedback': True,
            'uptake_classification': np.random.choice(['SUCCESSFUL', 'NO-UPTAKE', 'UNSUCCESSFUL'])
        })

df = pd.DataFrame(data)
df.to_csv('data/sample_experiment1.csv', index=False)
```

---

## Questions?

If you encounter data format issues:
1. Check this specification
2. Run the validation script: `python src/0_prepare_data.py`
3. Review error messages carefully
4. Contact: jieun_han@kaist.ac.kr
