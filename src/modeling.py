"""
Stage 3/4: guided dimensionality reduction, clustering, and topic representation.

The pipeline is written out explicitly instead of hidden inside BERTopic's
fit_transform, so each step can be cached, swept, and described in the paper:

1. (optional) Guided step. Same rule BERTopic uses for seed_topic_list: each
   chunk is matched to its most similar seed topic (or to the corpus mean, in
   which case it is left unlabeled); matched embeddings are nudged 25% toward
   the seed, and the seed labels are given to UMAP as a semi-supervised target.
2. UMAP to 5 dimensions.
3. KMeans with a fixed seed and n_init=10. No automatic topic merging
   (v1's nr_topics="auto" silently merged 50 clusters into 29).
4. BERTopic is used only for representations (c-TF-IDF keywords plus a
   KeyBERT-style aspect), fit on the cluster labels we already have.
   Topic assignments are the fitted labels; nothing is re-predicted with
   transform() (v1 did, which skipped the guided step).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import normalize

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Seed topics (Appendix D, de-duplicated)
# ---------------------------------------------------------------------------
# v1 repeated several terms across topics ("governance", "health financing",
# "vector control", "One Health", "sustainability planning"), which pulls the
# same chunk toward two seeds. Each term now lives in exactly one topic.

SEED_TOPICS: dict[str, list[str]] = {
    "HIV/AIDS": [
        "HIV/AIDS", "PEPFAR", "antiretroviral therapy", "viral suppression", "pre-exposure prophylaxis",
        "DREAMS initiative", "key populations", "95-95-95 targets", "test and treat", "AIDS-free generation",
        "UNAIDS", "HIV testing", "ART adherence", "prevention of mother-to-child transmission", "PMTCT",
    ],
    "Malaria": [
        "malaria", "President's Malaria Initiative", "PMI", "vector control", "insecticide-treated nets",
        "indoor residual spraying", "artemisinin-based combination therapy", "rapid diagnostic tests",
        "integrated vector management", "malaria elimination", "entomological monitoring",
        "antimalarial drug resistance", "seasonal malaria chemoprevention", "larviciding",
    ],
    "Maternal and child health": [
        "maternal health", "child health", "maternal mortality", "newborn care", "childhood immunization",
        "obstetric care", "kangaroo mother care", "integrated management of childhood illness", "IMCI",
        "antenatal care", "postnatal care", "skilled birth attendance", "emergency obstetric care",
        "maternal nutrition", "child survival",
    ],
    "Family planning and reproductive health": [
        "family planning", "contraceptive security", "reproductive health", "unmet need", "birth spacing",
        "LARC methods", "contraceptive prevalence", "method mix", "reproductive rights",
        "Demographic and Health Surveys", "total fertility rate", "UNFPA", "voluntary family planning",
        "adolescent reproductive health", "population dynamics", "contraceptive counseling",
        "unintended pregnancy",
    ],
    "Health systems strengthening": [
        "health systems strengthening", "health workforce", "human resources for health",
        "supply chain management", "health information systems", "DHIS2", "quality improvement",
        "universal health coverage", "UHC", "district health systems", "public financial management",
        "resilient health systems", "decentralization",
    ],
    "Global health security": [
        "global health security", "pandemic preparedness", "disease surveillance", "outbreak response",
        "International Health Regulations", "One Health approach", "zoonotic diseases",
        "antimicrobial resistance", "emergency operations centers", "biosafety", "biosecurity",
        "rapid response teams", "event-based surveillance", "GHSA", "emerging infectious diseases",
        "biological threats",
    ],
    "COVID-19": [
        "COVID-19", "SARS-CoV-2", "vaccine equity", "vaccine hesitancy", "oxygen supply",
        "COVID-19 therapeutics", "PPE", "testing capacity", "genomic sequencing", "COVAX facility",
        "pandemic response", "infection prevention and control", "social distancing",
        "vaccine distribution", "vaccine cold chain", "COVID variants",
    ],
    "Digital health": [
        "digital health", "mHealth", "telemedicine", "electronic medical records", "interoperability",
        "health management information systems", "geospatial analysis", "artificial intelligence",
        "digital technologies", "connectivity", "digital literacy", "data protection",
        "mobile applications", "remote monitoring", "digital transformation",
    ],
    "Nutrition": [
        "nutrition", "stunting", "wasting", "micronutrient deficiencies", "infant and young child feeding",
        "breastfeeding", "complementary feeding", "food fortification", "Scaling Up Nutrition",
        "SUN movement", "nutrition-sensitive agriculture", "first 1000 days", "anemia",
        "nutrition surveillance", "food security", "malnutrition treatment", "RUTF",
    ],
    "WASH": [
        "WASH", "water security", "sanitation", "hygiene", "water quality", "open defecation",
        "handwashing", "water treatment", "community-led total sanitation", "CLTS", "menstrual hygiene",
        "water point sustainability", "safely managed services", "WASH in healthcare facilities",
        "water access", "sanitation marketing",
    ],
    "Tuberculosis": [
        "tuberculosis", "TB", "drug-resistant tuberculosis", "MDR-TB", "DOTS", "GeneXpert",
        "TB case finding", "TB preventive therapy", "isoniazid preventive therapy", "TB diagnosis",
        "contact investigation", "TB/HIV co-infection", "treatment success rate", "latent TB infection",
        "TB infection control", "national TB programs",
    ],
    "Equity, gender, and vulnerable populations": [
        "health equity", "gender equality", "gender-based violence", "marginalized populations",
        "disability inclusion", "LGBTQI+ health", "social determinants of health", "indigenous health",
        "adolescent health", "women's empowerment", "gender analysis", "gender integration",
        "youth engagement", "inclusive programming", "intersectionality",
    ],
    "Private sector and health financing": [
        "private sector engagement", "public-private partnerships", "domestic resource mobilization",
        "health financing", "health insurance", "results-based financing", "social health insurance",
        "financial protection", "blended finance", "corporate social responsibility",
        "social entrepreneurship", "market-based solutions", "total market approach",
        "financial sustainability", "revenue generation",
    ],
    "Community health": [
        "community health workers", "CHWs", "community-based services", "task shifting",
        "community engagement", "community health strategy", "village health teams",
        "community case management", "community mobilization", "integrated community case management",
        "iCCM", "peer educators", "community health committees", "community participation",
        "last mile health services", "community health systems",
    ],
    "Health policy, governance, and leadership": [
        "health policy", "governance", "strategic planning", "health legislation", "regulatory frameworks",
        "accountability", "transparency", "policy dialogue", "stakeholder engagement",
        "evidence-based policymaking", "policy implementation", "health systems governance",
        "leadership development", "institutional capacity building", "country ownership",
        "sustainability planning", "transition planning",
    ],
    "Climate and environmental health": [
        "climate change", "environmental health", "climate resilience", "climate adaptation",
        "extreme weather events", "heat-related illness", "air pollution", "climate-sensitive diseases",
        "planetary health", "climate mitigation", "climate-smart healthcare", "environmental determinants",
        "climate risk assessment",
    ],
    "Neglected tropical diseases": [
        "neglected tropical diseases", "NTDs", "mass drug administration", "lymphatic filariasis",
        "schistosomiasis", "trachoma", "soil-transmitted helminths", "onchocerciasis", "disease mapping",
        "preventive chemotherapy", "morbidity management and disability prevention", "NTD elimination",
        "integrated NTD control",
    ],
    "NCDs and mental health": [
        "non-communicable diseases", "NCDs", "cardiovascular disease", "diabetes",
        "chronic respiratory diseases", "cancer", "mental health", "NCD prevention", "tobacco control",
        "alcohol use", "physical activity", "healthy diets", "mental health integration", "depression",
        "anxiety disorders", "psychological first aid", "trauma-informed care",
    ],
}


def _check_seed_overlap() -> None:
    seen: dict[str, str] = {}
    for topic, terms in SEED_TOPICS.items():
        for t in terms:
            key = t.lower()
            if key in seen:
                raise ValueError(f"Seed term '{t}' appears in both '{seen[key]}' and '{topic}'")
            seen[key] = topic


_check_seed_overlap()


# ---------------------------------------------------------------------------
# Vocabulary / stopwords (affect keywords only, never cluster assignment)
# ---------------------------------------------------------------------------

def get_stopwords() -> list[str]:
    """
    Single-token stopwords for the c-TF-IDF vocabulary.

    These change which keywords describe a topic. They do not change which
    topic a chunk is assigned to (that is decided on embeddings). Multi-word
    entries are omitted because CountVectorizer ignores them. Bare numbers are
    excluded by the token pattern instead of being listed.
    """
    publication = """prepared publication suggested citation isbn report reports washington dc
        official views positions contents executive summary background overview methodology
        appendix annex page pages section chapter paragraph figure table reference references
        bibliography footnote note volume https www usaid gov org pdf disclaimer""".split()
    time_terms = """january february march april may june july august september october november
        december jan feb mar apr jun jul aug sep sept oct nov dec month months monthly quarter
        quarters quarterly annual annually year years fiscal fy""".split()
    time_terms += [f"fy{y}" for y in range(10, 30)] + [f"fy20{y}" for y in range(10, 30)]
    fragments = """de la les en des et le du para dan pa za ke tes wa ts ee tr ea ti te ie io ro sb
        fn aa uo ye na tbd ii iii iv vi vii viii ix xi xii""".split()
    # v1 also removed "budget", "procurement", "outreach", "implemented": those carry policy
    # meaning and are kept now. reduce_frequent_words=True handles generic high-frequency terms.
    generic = """total number percent frequency details questionnaire respondents""".split()
    return sorted(set(ENGLISH_STOP_WORDS) | set(publication) | set(time_terms) | set(fragments) | set(generic))


def make_vectorizer(ngram_range=cfg.VECTORIZER_NGRAM_RANGE, binary=False, min_df=cfg.VECTORIZER_MIN_DF,
                    max_df=cfg.VECTORIZER_MAX_DF):
    """
    min_df/max_df here count *chunks* (TermMatrices fits on chunks).

    Careful: BERTopic fits its vectorizer on one concatenated document per
    topic, so inside BERTopic min_df counts *topics*. v1 passed min_df=20 to
    BERTopic, which kept only words that appear in at least 20 topics and
    threw away the distinctive ones. The BERTopic-side vectorizer below uses
    min_df=1, max_df=1.0 for that reason.
    """
    return CountVectorizer(
        stop_words=get_stopwords(),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        token_pattern=cfg.VECTORIZER_TOKEN_PATTERN,
        lowercase=True,
        binary=binary,
    )


# ---------------------------------------------------------------------------
# Guided step + UMAP
# ---------------------------------------------------------------------------

def guided_labels(embeddings: np.ndarray, embedding_model) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce BERTopic's guided-topic rule explicitly.

    Returns (y, nudged_embeddings). y[i] = seed index, or -1 when the chunk is
    closer to the corpus mean than to any seed (unlabeled for UMAP).
    """
    seed_texts = [", ".join(terms) for terms in SEED_TOPICS.values()]
    seed_emb = embedding_model.encode(seed_texts, normalize_embeddings=True, convert_to_numpy=True)
    targets = np.vstack([seed_emb, embeddings.mean(axis=0, keepdims=True)])
    targets = normalize(targets)
    sims = embeddings @ targets.T  # embeddings are already L2-normalised
    y = sims.argmax(axis=1)
    y[y == len(seed_texts)] = -1

    nudged = embeddings.copy()
    for s in range(len(seed_texts)):
        m = y == s
        nudged[m] = 0.75 * nudged[m] + 0.25 * seed_emb[s]
    return y.astype(int), nudged


def reduce_dimensions(embeddings: np.ndarray, y: np.ndarray | None, seed: int) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=cfg.UMAP_N_NEIGHBORS,
        n_components=cfg.UMAP_N_COMPONENTS,
        min_dist=cfg.UMAP_MIN_DIST,
        metric=cfg.UMAP_METRIC,
        random_state=seed,
        low_memory=True,
        verbose=True,
    )
    n = len(embeddings)
    if cfg.UMAP_FIT_SAMPLE and n > cfg.UMAP_FIT_SAMPLE:
        idx = np.random.default_rng(seed).choice(n, cfg.UMAP_FIT_SAMPLE, replace=False)
        reducer.fit(embeddings[idx], y=None if y is None else y[idx])
        logger.info("UMAP fit on %d sampled chunks; transforming all %d ...", len(idx), n)
        out = np.empty((n, cfg.UMAP_N_COMPONENTS), dtype=np.float32)
        step = 50_000
        for start in range(0, n, step):
            out[start:start + step] = reducer.transform(embeddings[start:start + step])
        return out
    return reducer.fit_transform(embeddings, y=y)


def get_reduced(data_dir: Path, embeddings: np.ndarray, seed: int, seeded: bool, embedding_model=None) -> np.ndarray:
    """UMAP output for (seed, seeded), cached on disk."""
    cache = data_dir / "umap_cache"
    cache.mkdir(exist_ok=True)
    path = cache / (f"{'seeded' if seeded else 'unseeded'}_seed{seed}_nn{cfg.UMAP_N_NEIGHBORS}"
                    f"_d{cfg.UMAP_N_COMPONENTS}_fit{cfg.UMAP_FIT_SAMPLE or 'all'}.npy")
    if path.exists():
        red = np.load(path)
        if red.shape[0] == embeddings.shape[0]:
            logger.info("Loaded cached UMAP %s", path.name)
            return red
    if seeded:
        if embedding_model is None:
            from src.embedding import load_embedding_model

            embedding_model = load_embedding_model()
        y, emb_in = guided_labels(embeddings, embedding_model)
        logger.info(
            "Guided step: %.1f%% of chunks matched a seed topic",
            100 * (y >= 0).mean(),
        )
        np.save(cache / "guided_seed_labels.npy", y)
    else:
        y, emb_in = None, embeddings
    logger.info("Fitting UMAP (seed=%d, seeded=%s) on %d x %d ...", seed, seeded, *emb_in.shape)
    red = reduce_dimensions(emb_in, y, seed)
    np.save(path, red)
    return red


def cluster(reduced: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=cfg.KMEANS_N_INIT, random_state=seed).fit_predict(reduced)


# ---------------------------------------------------------------------------
# Fast c-TF-IDF + coherence (used by the sweep and the final report)
# ---------------------------------------------------------------------------

class TermMatrices:
    """Count matrices built once and reused for every (k, seed) in the sweep."""

    def __init__(self, texts: list[str]):
        self.vec = make_vectorizer()
        self.X = self.vec.fit_transform(texts).tocsr()
        self.vocab = np.array(self.vec.get_feature_names_out())
        self.is_unigram = np.array([" " not in w for w in self.vocab])
        # Binary unigram document-term matrix for NPMI
        self.Xb = (self.X[:, np.where(self.is_unigram)[0]] > 0).astype(np.float32).tocsc()
        self.uni_index = {w: i for i, w in enumerate(self.vocab[self.is_unigram])}
        self.doc_freq = np.asarray(self.Xb.sum(axis=0)).ravel()
        self.n_docs = self.Xb.shape[0]
        logger.info("Vocabulary: %d terms (%d unigrams)", len(self.vocab), self.is_unigram.sum())

    def ctfidf(self, labels: np.ndarray, k: int) -> np.ndarray:
        """BERTopic's c-TF-IDF with reduce_frequent_words=True. Returns dense (k x V)."""
        onehot = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels)))
        T = (onehot @ self.X).astype(np.float64)
        df = np.asarray(T.sum(axis=0)).ravel()
        avg = T.sum() / k
        idf = np.log(avg / np.maximum(df, 1) + 1)
        T = normalize(T, norm="l1", axis=1)
        T.data = np.sqrt(T.data)
        return (T @ sp.diags(idf)).toarray()

    def top_words(self, scores: np.ndarray, n: int = cfg.N_TOP_WORDS, unigram_only: bool = False) -> list[list[str]]:
        out = []
        for row in scores:
            order = np.argsort(-row)
            if unigram_only:
                order = order[self.is_unigram[order]]
            out.append([self.vocab[i] for i in order[:n] if row[i] > 0])
        return out

    def npmi(self, words: list[str]) -> float:
        idx = [self.uni_index[w] for w in words if w in self.uni_index]
        if len(idx) < 2:
            return np.nan
        sub = self.Xb[:, idx]
        co = (sub.T @ sub).toarray() / self.n_docs
        p = np.diag(co).copy()
        vals = []
        eps = 1e-12
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pij = co[i, j]
                if pij <= 0:
                    vals.append(-1.0)
                    continue
                pmi = np.log((pij + eps) / (p[i] * p[j] + eps))
                vals.append(pmi / -np.log(pij + eps))
        return float(np.mean(vals))


def topic_diversity(word_lists: list[list[str]]) -> float:
    words = [w for ws in word_lists for w in ws]
    return len(set(words)) / max(len(words), 1)


def concentration(labels: np.ndarray, doc_ids: np.ndarray, k: int) -> pd.DataFrame:
    """Per topic: chunks, distinct documents, and share supplied by the single largest document."""
    df = pd.DataFrame({"topic": labels, "doc_id": doc_ids})
    g = df.groupby(["topic", "doc_id"]).size().rename("n").reset_index()
    rows = []
    for t in range(k):
        sub = g[g["topic"] == t]
        n = sub["n"].sum()
        rows.append({
            "topic": t,
            "n_chunks": int(n),
            "n_docs": int(len(sub)),
            "top_doc_id": int(sub.loc[sub["n"].idxmax(), "doc_id"]) if n else -1,
            "top_doc_share": float(sub["n"].max() / n) if n else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def fit_representations(texts: list[str], embeddings: np.ndarray, labels: np.ndarray, embedding_model):
    """
    Fit BERTopic on precomputed cluster labels, only to get topic representations.
    Main = c-TF-IDF keywords; KeyBERT = embedding-based keyword aspect.
    """
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        vectorizer_model=make_vectorizer(min_df=1, max_df=1.0),  # topic-level counts; see make_vectorizer
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model={"KeyBERT": KeyBERTInspired(random_state=cfg.RANDOM_SEED)},
        top_n_words=cfg.N_TOP_WORDS,
        nr_topics=None,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_model.fit(texts, embeddings=embeddings, y=labels)
    return topic_model


LABEL_SYSTEM = (
    "You label topics from a topic model of USAID global health documents. "
    "Return a short, literal label (at most 6 words) describing what the passages are about. "
    "Name a country or program only if nearly all passages share it. "
    "Do not infer policy intent or significance. Return only the label."
)


def gpt_labels(topic_model, n_docs: int = 5) -> dict[int, str]:
    """Optional human-readable labels. Display only: never used for any metric."""
    if not cfg.OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set; skipping GPT labels.")
        return {}
    import openai

    client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
    labels = {}
    rep_docs = topic_model.get_representative_docs()
    for t in sorted(topic_model.get_topics()):
        if t == -1:
            continue
        words = ", ".join(w for w, _ in topic_model.get_topic(t))
        docs = "\n---\n".join(d[:800] for d in rep_docs.get(t, [])[:n_docs])
        resp = client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": LABEL_SYSTEM},
                {"role": "user", "content": f"Keywords: {words}\n\nPassages:\n{docs}"},
            ],
        )
        labels[t] = resp.choices[0].message.content.strip().strip('"')
        logger.info("Topic %d: %s", t, labels[t])
    return labels


def save_final(out_dir: Path, topic_model, labels: np.ndarray, chunks: pd.DataFrame, docs: pd.DataFrame,
               terms: TermMatrices, k: int, gpt: dict[int, str]) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"chunk_id": chunks["chunk_id"], "doc_id": chunks["doc_id"], "topic": labels}).to_csv(
        out_dir / "topics.csv", index=False
    )

    scores = terms.ctfidf(labels, k)
    words = terms.top_words(scores)
    uni = terms.top_words(scores, unigram_only=True)
    conc = concentration(labels, chunks["doc_id"].to_numpy(), k)
    fname = docs.set_index("doc_id")["filename"]

    keybert = {t: [w for w, _ in v] for t, v in topic_model.topic_aspects_.get("KeyBERT", {}).items()}
    info = conc.copy()
    info["keywords"] = [", ".join(w) for w in words]
    info["keybert"] = [", ".join(keybert.get(t, [])) for t in range(k)]
    info["npmi"] = [terms.npmi(u) for u in uni]
    info["gpt_label"] = [gpt.get(t, "") for t in range(k)]
    info["top_doc_filename"] = info["top_doc_id"].map(fname)
    info["single_doc_dominated"] = info["top_doc_share"] > cfg.SINGLE_DOC_DOMINANCE
    info["share_of_chunks"] = info["n_chunks"] / info["n_chunks"].sum()
    info.to_csv(out_dir / "topic_info.csv", index=False)

    topic_model.save(
        str(out_dir / "bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=cfg.EMBEDDING_MODEL_NAME,
    )
    (out_dir / "seed_topics.json").write_text(json.dumps(SEED_TOPICS, indent=2))
    return info
