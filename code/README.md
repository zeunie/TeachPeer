# TeachPeer code

Analysis code accompanying the manuscript *Human Explanatory Behaviour with an Imperfect LLM Partner* (Nature Communications).

The anonymized dataset is in the sibling folder `TeachPeer_dataset`. Copy those CSV files into `data/` or keep the two folders next to each other.

## Layout

```
paper_submission_code/
├── src/
│   ├── 0_prepare_data.py             # Data preparation and validation
│   ├── 1_baseline_llm_simulation.py  # Experiment 1: baseline LLM simulation
│   ├── 2_llm_peer_generation.py      # LLM-peer essay generation and validation
│   ├── 3_statistical_analysis.py     # Experiment 2: statistical analysis
│   ├── utils.py                      # TOST, KS, Cramér's V, text features
│   └── run_glmm_in_R.R               # Binomial GLMMs (lme4)
├── data/                             # Add the released CSV files here
├── outputs/
│   ├── figures/
│   └── results/
├── run_all_analyses.py
├── config.example.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- R 4.2+ with `lme4` and `dplyr` (needed only for the mixed-effects models)

```bash
pip install -r requirements.txt
Rscript -e "install.packages(c('lme4', 'dplyr'), repos='https://cloud.r-project.org')"
```

## Reproduce the analyses

From this folder:

```bash
python run_all_analyses.py
```

Individual steps:

```bash
python src/0_prepare_data.py
python src/1_baseline_llm_simulation.py
python src/2_llm_peer_generation.py
python src/3_statistical_analysis.py
```

The default pipeline uses the released labels and essays. It does not require an API key.

Optional regeneration:

```bash
cp config.example.py config.py   # then add an API key
python src/1_baseline_llm_simulation.py --simulate
python src/2_llm_peer_generation.py --generate --essay "..." --topic "cooking at home"
```

Exact JSON schemas for the classroom peer-generation protocol are in Supplementary Information, Section 4. The manuscript range-validation figure used 164 generation pairs; `experiment_2_essay.csv` contains the 171 classroom-session essays.

## Expected results

- Human edit rate: 54.28% no edit, 45.72% edit
- Human uptake: 17.12% successful, 70.49% no uptake, 12.39% unsuccessful
- Human vs pooled LLM edit: χ²(1) = 1331.56, *p* < .001
- LLM uptake homogeneity: χ²(6) = 10.14, *p* = .119
- Amplification χ²(2) = 165.12; Constructive χ²(2) = 207.11
- Extraneous load: *F*(2, 54) = 7.316, *p* = .002
- Essay improvement: *F*(2, 54) = 2.564, *p* = .086, η² = .087

## Citation

Han, J. et al. Human Explanatory Behaviour with an Imperfect LLM Partner.
