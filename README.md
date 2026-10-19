# Bridging the Divide: USAID Global Health Priorities Under Trump and Biden

Topic-modeling pipeline for comparing USAID global health documents from the first Trump administration (2017-01-20 to 2021-01-19) and the Biden administration (2021-01-20 to 2024-10-23, when the corpus was collected), drawn from the USAID Development Experience Clearinghouse (DEC).

- `legacy/v1/` is the code behind the March 2025 Signature Work thesis, kept unchanged so those results can be reproduced.
- Everything else is the revised **v2** pipeline used for the preprint.

## What changed in v2 and why

| Issue in v1 | v2 |
|---|---|
| PDF text split on blank lines, which PyMuPDF rarely emits, so units were roughly whole pages | Paragraphs built from PyMuPDF text blocks; running headers and footers removed per document |
| `paraphrase-MiniLM-L6-v2` truncates at 128 tokens, so it saw only the top of each page | `all-mpnet-base-v2` (384 tokens), with paragraphs chunked to ≤180 words; `02_embed.py` reports any truncation |
| `nr_topics="auto"` silently merged 50 KMeans clusters into 29 (including one 60% topic) | No automatic merging. K is chosen from a stability sweep before any administration comparison |
| KMeans without a seed and run once | Fixed seeds, `n_init=10`, stability measured across seeds (ARI) |
| Topic shares came from `transform()`, which skips the guided step used in fitting | The fitted cluster labels are the assignments |
| Shares and permutation tests used paragraphs (pages) as independent units | Shares are computed per document (each document weighted equally); permutation is by document; BH-FDR correction |
| Coherence was computed on the words of GPT-generated labels, over the first 5,000 rows | NPMI on top c-TF-IDF keywords over the full corpus, plus diversity and single-document-topic counts |
| BERTopic vectorizer `min_df=20` counted topics, not paragraphs, so keywords had to appear in ≥20 topics | Keywords from c-TF-IDF over chunk-level counts; BERTopic's own vectorizer uses `min_df=1` |
| Stopwords described as reducing clustering noise | Stopwords affect keywords only (documented); multi-word stopwords that sklearn ignored are removed |
| References filter dropped any paragraph starting "However,", "Overall,", and so on; all parenthetical text deleted (including acronyms) | Reference entries detected by citation signals; parentheses kept |
| Year-only administration labels (Jan 1-19, 2021 counted as Biden) | Full dates from filenames |
| No controls for document type or country | Within-document-type permutation and covariate regressions when DEC metadata is supplied |
| No lag check | Robustness check excluding each term's first 365 days (documents designed under the predecessor) |
| Mean cosine similarity with no null distribution | Cross-validated classifier AUC against a permutation null |
| Topic interpretation from LLM labels | Labels are optional and display-only; `reading_samples.csv` has passages to read for every topic |

## Running it

```bash
pip install -r requirements.txt

# 1. PDFs -> chunks.csv, documents.csv, manifest.csv (MD5 per file), filter_counts.csv
python scripts/01_extract_text.py --pdf-dir /path/to/pdfs --data-dir data [--metadata data/dec_metadata.csv]

# 2. Embed once (cached to data/embeddings.npy). A GPU or Apple Silicon (MPS) helps a lot.
python scripts/02_embed.py --data-dir data

# 3. Choose K. Look at outputs/sweep/seeded/sweep_summary.csv and write down the reason
#    BEFORE stage 5. Nothing in the sweep looks at administration differences.
python scripts/03_sweep.py --data-dir data
python scripts/03_sweep.py --data-dir data --no-seeds        # robustness: unguided model

# 4. Final model at the chosen K
python scripts/04_train_model.py --data-dir data --k 40 --k-rationale "highest seed ARI with <2 single-doc topics" [--gpt-labels]

# 5. Administration comparison and robustness checks
python scripts/05_run_analysis.py --data-dir data --model-dir outputs/final_k40_seeded
```

The DEC metadata CSV is optional. Column names are auto-detected; otherwise set `META_FILENAME_COL`, `META_DOCTYPE_COL` and `META_COUNTRY_COL` in `.env`.

## Main outputs (`outputs/final_k*/analysis/`)

- `results_summary.md`: corpus counts, the largest differences with CIs and q-values under each robustness check, separability, and term-mention rates
- `administration_comparison.csv`: every topic, with document-weighted and chunk-weighted prevalence, the Biden−Trump difference, bootstrap CI, permutation p/q, within-document-type and lag-robust p/q, regression coefficient, number of documents, and the single largest document's share
- `administration_differences.png`, `yearly_prevalence.png`
- `reading_samples.csv`: up to 15 passages per topic from distinct documents. Read these before interpreting a topic.
- `mentions_by_year.csv` / `mentions_by_administration.csv`: share of documents mentioning COVID, AMR, family planning, Mexico City Policy/PLGHA, multilateral partners, and so on
- `run_info.json` in every output folder: full config, package versions, and git commit

## Data availability

The DEC went offline in 2025. `data/manifest.csv` lists every PDF with its MD5 hash and processing status. Deposit it, together with the DEC export, on OSF or Zenodo so the corpus can be reconstructed.

## Author

William (Liam) Powell. Originally a Signature Work Product, Duke Kunshan University, advised by Annemieke van den Dool.
