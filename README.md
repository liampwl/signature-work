# Bridging the Divide: Global Health and International Development Policy Priorities in the Biden and Trump Eras

This repository contains the analysis pipeline for a research project examining policy shifts in global health and international development using **BERTopic** topic modeling. The pipeline processes USAID policy documents (PDFs), trains a guided topic model with OpenAI-based labeling, and runs a suite of statistical analyses to compare policy attention across U.S. administrations.

## Repository Structure

```
├── config.py                        # Central configuration (paths, hyperparameters, API keys)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Template for environment variables
│
├── src/                             # Core library modules
│   ├── preprocessing.py             # PDF extraction (OCR), text cleaning, stopwords
│   ├── modeling.py                  # BERTopic model building, seed topics, training
│   ├── utils.py                     # Shared helpers (logging, topic labels)
│   └── analysis/                    # Analysis sub-package
│       ├── administration.py        # Trump vs. Biden topic comparison & null-model testing
│       ├── coherence.py             # Topic coherence metrics (c_v, u_mass)
│       ├── temporal.py              # Temporal granularity, topic evolution, attention shifts
│       ├── similarity.py            # Document embedding similarity (t-SNE, cosine)
│       ├── events.py                # Focusing event detection & punctuated equilibrium
│       └── reporting.py             # Methodological summary report generation
│
├── scripts/                         # CLI entry points (run in order)
│   ├── 01_extract_text.py           # Stage 1: OCR → CSV
│   ├── 02_train_model.py            # Stage 2: Train & save BERTopic model
│   └── 03_run_analysis.py           # Stage 3: Run all analyses on saved model
│
└── docs/
    └── wnp2_PowellIII_2025_vandenDool_SWPaper.pdf   # Published paper
```

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/<your-username>/signature-work.git
cd signature-work
pip install -r requirements.txt

# 2. Set up your environment variables
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY

# 3. Run the pipeline
python scripts/01_extract_text.py --pdf-dir ./data/pdfs --output ./data/processed_paragraphs.csv
python scripts/02_train_model.py --csv ./data/processed_paragraphs.csv --output-dir ./outputs/run_001
python scripts/03_run_analysis.py --csv ./data/processed_paragraphs.csv --model-dir ./outputs/run_001/topic_model --output-dir ./outputs/run_001
```

Each script supports `--help` for a full list of options.

## About the Project

This research investigates how **USAID's global health policy priorities** evolved under the Biden and Trump administrations. Using **BERTopic** with guided seed topics across 18 global health domains, the project identifies key themes, measures policy attention shifts, and applies theoretical frameworks including punctuated equilibrium theory and Kingdon's policy windows model.

## Author

**William Powell**
*Signature Work Product, in partial fulfillment of the Duke Kunshan University Undergraduate Degree Program*

**Duke Kunshan University** | *March 9, 2025*
**arXiv preprint:** *TKTK (pending publication)*
