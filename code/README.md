# Human Explanatory Behaviour with Imperfect LLM Partner

This repository contains reproducible analysis code for the paper submitted to Nature Communications.

**Authors**: Jieun Han, Junyeong Park, Seohyun Park, Haneul Yoo, Hyoungwook Jin, Xing Xie, Evelyne Viegas, Sean Rintel, Miran Lee, Alice Oh, So-Yeon Ahn, Fangzhao Wu

## Repository Structure

```
paper_submission_code/
├── src/                              # Analysis scripts
│   ├── 0_prepare_data.py            # Data preparation and validation
│   ├── 1_baseline_llm_simulation.py # Experiment 1: Baseline LLM simulation
│   ├── 2_llm_peer_generation.py     # LLM peer essay generation & validation
│   ├── 3_statistical_analysis.py    # Experiment 2: Statistical analysis
│   └── utils.py                      # Utility functions (TOST, KS test, etc.)
├── data/                             # Data directory (add your data files here)
├── outputs/                          # Analysis outputs
│   ├── figures/                      # Generated figures (PDF format)
│   └── results/                      # Statistical results (CSV format)
├── run_all_analyses.py               # Master script to run complete pipeline
├── config.example.py                 # Configuration template
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Requirements

- Python 3.10 or higher
- OpenAI API key (for GPT-4o and GPT-4o-mini)

### Python Packages

```bash
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openai>=1.0.0
tqdm>=4.65.0
```

## Installation

1. **Clone or download this repository**

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Configure API keys**:
```bash
cp config.example.py config.py
# Edit config.py and add your OpenAI API key
```

4. **Prepare your data**:
   - Place raw data files in the `data/` directory
   - Expected files:
     - `0821_simple_df.csv` (for Experiment 1)
     - `experiment2_raw.csv` (for Experiment 2)

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Run all analyses in sequence:

```bash
python run_all_analyses.py
```

This will execute:
1. Data preparation and validation
2. Baseline LLM simulation (Experiment 1)
3. LLM peer essay generation and validation
4. Statistical analysis (Experiment 2)

### Option 2: Run Individual Analyses

#### Step 0: Prepare Data
```bash
python src/0_prepare_data.py
```
- Validates input data
- Creates processed datasets
- Generates data summary report

#### Step 1: Baseline LLM Simulation (Experiment 1)
```bash
python src/1_baseline_llm_simulation.py
```
- Simulates student responses using baseline LLMs
- Classifies uptake patterns (SUCCESSFUL/UNSUCCESSFUL/NO-UPTAKE)
- Compares real learners vs. baseline LLMs
- **Statistical tests**: Chi-square, Cramér's V

#### Step 2: LLM Peer Essay Generation
```bash
python src/2_llm_peer_generation.py
```
- Extracts error patterns from learner essays
- Generates LLM peer essays with mirrored errors
- Validates using TOST equivalence testing
- **Analyses**: KC count, word count, TTR, FRES

#### Step 3: Statistical Analysis (Experiment 2)
```bash
python src/3_statistical_analysis.py
```
- K-means clustering of uptake profiles
- Instructional strategy analysis
- ICAP engagement classification
- Cognitive load analysis (ANOVA)

## Key Analyses & Statistical Methods

### Experiment 1: Real Learner vs. Baseline LLM Behavior (N=115)

**Research Question**: Do current LLMs exhibit human-like feedback negotiation?

**Analyses**:
- **Edit behavior**: Chi-square test (χ²) comparing editing rates
- **Uptake classification**: GPT-4o automated labeling
- **Cross-model comparison**: Homogeneity testing (GPT-4o, GPT-4o-mini, Llama-3.1-70B, Claude-3.5-Sonnet)
- **Human-LLM divergence**: Effect size quantification (Cramér's V)

**Key Finding**: Baseline LLMs show 73-76% successful uptake vs. 17% in real learners (χ²(2) = 687.51, p < .001, V = .501)

### LLM Peer Generation & Validation (N=164 pairs)

**Research Question**: Can we create learner-like LLM peers through error mirroring?

**Validation Methods**:
1. **Equivalence testing (TOST)**: Margins = ±0.5 SD of human distribution
2. **Distribution comparison**: Kolmogorov-Smirnov tests
3. **Distance metrics**: Wasserstein distance (normalized), Jensen-Shannon divergence
4. **Range validation**: Bootstrap confidence intervals (5,000 resamples)

**Features Analyzed**:
- Knowledge Component (KC) error count
- Word count
- Type-Token Ratio (TTR)
- Flesch Reading Ease Score (FRES)

**Key Finding**: KC errors achieved equivalence (p < .001), with 92.7-100% of LLM values within human range

### Experiment 2: Uptake Clustering & Engagement Analysis (N=57, 171 sessions)

**Research Question**: How does peer uptake behavior shape explanatory engagement?

**Clustering Analysis**:
- **Method**: K-means clustering (k=3) on uptake proportions
- **Profiles identified**:
  - Low-uptake: 100% no-uptake
  - High-uptake: 71.4% successful, 14.3% unsuccessful, 14.3% no-uptake
  - Mixed-uptake: 44.4% no-uptake, 33.3% unsuccessful, 22.2% successful

**Instructional Strategy Analysis**:
- **Coding scheme**: 8 strategies (Reformulate, Amplification, Reject, Accept, etc.)
- **Statistics**: Chi-square tests + mixed-effects models
- **Key result**: Amplification higher in low-uptake (53.63%) vs. high-uptake (24.94%), χ²(2) = 165.12, p < .001

**ICAP Framework**:
- **Levels**: Constructive, Active, Passive
- **Key result**: Low-uptake elicited higher constructive engagement (χ²(2) = 210.14, p < .001)

**Cognitive Load**:
- **Method**: One-way ANOVA on intrinsic, extraneous, germane load
- **Key result**: Extraneous load higher in low-uptake, F(2,54) = 7.316, p = .002

## Output Files

### Results (CSV)
- `baseline_llm_simulation_results.csv`: Uptake classifications from baseline LLMs
- `llm_peer_essays.csv`: Generated peer essays with features
- `equivalence_test_results.csv`: TOST and distribution comparison statistics
- `uptake_profiles_clustered.csv`: K-means clustering results
- `strategy_analysis.csv`: Instructional strategy distributions
- `icap_analysis.csv`: ICAP engagement distributions
- `cognitive_load_analysis.csv`: ANOVA results

### Figures (PDF, 300 DPI)
- `distribution_comparison.pdf`: Box plots + density plots (Fig. 2 in paper)
- `tost_equivalence.pdf`: TOST results visualization (Supplementary Fig. X)
- `uptake_clusters.pdf`: 3D cluster visualization (Fig. 3)
- `strategy_distribution.pdf`: Stacked bar chart (Fig. 4 left)
- `icap_engagement.pdf`: Grouped bar chart (Fig. 4 right)

## Reproducibility Notes

### Random Seeds
All random operations use `RANDOM_SEED = 42` (configurable in `config.py`)

### API Calls
- Temperature = 0 for all classification tasks
- Temperature = 0.7 for essay generation
- Max concurrent requests = 10 (rate limiting)

### Statistical Thresholds
- Equivalence margin: ±0.5 × SD
- Significance level: α = 0.05
- Bonferroni correction applied for post-hoc tests

## Data Availability

The full dataset is publicly available at: https://github.com/zeunie/TeachPeer

**Dataset includes**:
- 115 learner interaction logs (Experiment 1)
- 171 learning-by-teaching sessions (Experiment 2)
- 7,123 conversational turns
- Annotated instructional strategies (κ = 0.7685)

## Citation

```bibtex
[Citation will be added upon publication]
```

## License

[Add license information]

## Contact

For questions about the code or analyses:
- Jieun Han: jieun_han@kaist.ac.kr
- Fangzhao Wu: fangzwu@microsoft.com

## Acknowledgments

This research was supported by:
- MSIT (Ministry of Science, ICT), Korea - Global Research Support Program (RS-2024-00436680)
- Microsoft Research Asia
