"""
BERTopic model training with guided seed topics and OpenAI-based labeling.

Encapsulates all model configuration, seed topic definitions, and the
fit-transform pipeline (stage 2).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

import config as cfg
from src.preprocessing import get_enhanced_stopwords
from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Seed topics -- domain knowledge for guided modeling
# ---------------------------------------------------------------------------

SEED_TOPICS: list[list[str]] = [
    # 1. HIV/AIDS Programs and Implementation
    ["HIV/AIDS", "PEPFAR", "antiretroviral therapy", "viral suppression", "pre-exposure prophylaxis",
     "DREAMS initiative", "key populations", "90-90-90 targets", "test and treat", "AIDS-free generation",
     "UNAIDS", "HIV testing", "ART adherence", "prevention of mother-to-child transmission", "PMTCT"],
    # 2. Malaria and Vector-Borne Disease Control
    ["malaria", "President's Malaria Initiative", "PMI", "vector control", "insecticide-treated nets",
     "indoor residual spraying", "artemisinin-based combination therapy", "rapid diagnostic tests",
     "integrated vector management", "malaria elimination", "surveillance systems", "entomological monitoring",
     "drug resistance", "seasonal malaria chemoprevention", "larviciding"],

    # 3. Maternal and Child Health
    ["maternal health", "child health", "maternal mortality", "newborn care", "childhood immunization",
     "obstetric care", "kangaroo mother care", "integrated management of childhood illness", "IMCI",
     "antenatal care", "postnatal care", "skilled birth attendance", "emergency obstetric care",
     "maternal nutrition", "child survival"],

    # 4. Reproductive Health and Family Planning
    ["family planning", "contraceptive security", "reproductive health", "unmet need", "birth spacing",
     "LARC methods", "contraceptive prevalence", "method mix", "reproductive rights",
     "Demographic Health Surveys",
     "total fertility rate", "UNFPA", "voluntary family planning", "adolescent reproductive health",
     "population dynamics", "contraceptive counseling", "unintended pregnancy"],

    # 5. Health Systems Strengthening
    ["health systems strengthening", "governance", "health workforce", "human resources for health",
     "supply chain management", "health information systems", "DHIS2", "health financing",
     "quality improvement",
     "universal health coverage", "UHC", "district health systems", "public financial management",
     "community health systems", "resilient health systems", "decentralization"],

    # 6. Global Health Security and Emerging Threats
    ["global health security", "pandemic preparedness", "disease surveillance", "outbreak response",
     "International Health Regulations", "One Health approach", "zoonotic diseases", "antimicrobial resistance",
     "emergency operations centers", "biosafety", "biosecurity", "rapid response teams",
     "event-based surveillance",
     "GHSA", "emerging infectious diseases", "biological threats"],

    # 7. COVID-19 Response and Recovery
    ["COVID-19", "SARS-CoV-2", "vaccine equity", "vaccine hesitancy", "health system resilience",
     "oxygen supply", "therapeutic treatments", "PPE", "testing capacity", "genomic sequencing",
     "COVAX facility", "pandemic response", "infection prevention control", "social distancing",
     "vaccine distribution", "vaccine cold chain", "COVID variants"],

    # 8. Digital Health and Innovation
    ["digital health", "mHealth", "telemedicine", "electronic medical records", "interoperability",
     "health management information systems", "geospatial analysis", "artificial intelligence",
     "digital technologies", "connectivity", "digital literacy", "data protection", "digital solutions",
     "mobile applications", "remote monitoring", "digital transformation"],

    # 9. Nutrition and Food Security
    ["nutrition", "stunting", "wasting", "micronutrient deficiencies", "infant and young child feeding",
     "breastfeeding", "complementary feeding", "food fortification", "Scaling Up Nutrition", "SUN movement",
     "nutrition-sensitive agriculture", "first 1000 days", "anemia", "nutrition surveillance",
     "food security", "malnutrition treatment", "RUTF"],

    # 10. Water, Sanitation, and Hygiene
    ["WASH", "water security", "sanitation", "hygiene", "water quality", "open defecation",
     "handwashing", "water treatment", "community-led total sanitation", "CLTS", "menstrual hygiene",
     "water point sustainability", "safely managed services", "WASH in healthcare facilities",
     "water resource management", "water access", "sanitation marketing"],

    # 11. Tuberculosis Prevention and Control
    ["tuberculosis", "TB", "drug-resistant tuberculosis", "MDR-TB", "DOTS", "GeneXpert", "TB case finding",
     "TB preventive therapy", "isoniazid preventive therapy", "TB diagnosis", "contact tracing",
     "TB/HIV co-infection",
     "treatment success rate", "latent TB infection", "TB infection control", "national TB programs"],

    # 12. Health Equity, Gender, and Vulnerable Populations
    ["health equity", "gender equality", "gender-based violence", "marginalized populations",
     "disability inclusion",
     "LGBTQ+ health", "social determinants of health", "indigenous health", "adolescent health",
     "women's empowerment",
     "gender analysis", "gender integration", "youth engagement", "equity analysis", "inclusive programming",
     "intersectionality"],

    # 13. Private Sector Engagement and Health Financing
    ["private sector engagement", "public-private partnerships", "domestic resource mobilization",
     "health financing",
     "health insurance", "results-based financing", "social health insurance", "financial protection",
     "blended finance",
     "corporate social responsibility", "social entrepreneurship", "market-based solutions",
     "sustainability planning",
     "total market approach", "financial sustainability", "revenue generation"],

    # 14. Community Health Systems and Community Health Workers
    ["community health workers", "CHWs", "community-based services", "task shifting", "community engagement",
     "community health strategy", "village health teams", "community case management", "community mobilization",
     "community platforms", "integrated community case management", "iCCM", "peer educators",
     "community health committees",
     "community participation", "last mile health services", "CHW incentives"],

    # 15. Health Policy, Governance and Leadership
    ["health policy", "governance", "strategic planning", "health legislation", "regulatory frameworks",
     "accountability", "transparency", "policy dialogue", "stakeholder engagement",
     "evidence-based policymaking",
     "policy implementation", "health systems governance", "leadership development",
     "institutional capacity building",
     "country ownership", "sustainability planning", "transition planning"],

    # 16. Climate Change and Environmental Health
    ["climate change", "environmental health", "climate resilience", "climate adaptation",
     "vector distribution",
     "extreme weather events", "heat-related illness", "air pollution", "climate-sensitive diseases",
     "planetary health", "climate vulnerabilities", "climate mitigation", "climate-smart healthcare",
     "environmental determinants", "eco-health", "climate risk assessment", "One Health"],

    # 17. Neglected Tropical Diseases
    ["neglected tropical diseases", "NTDs", "mass drug administration", "lymphatic filariasis",
     "schistosomiasis",
     "trachoma", "soil-transmitted helminths", "onchocerciasis", "disease mapping", "preventive chemotherapy",
     "morbidity management", "disability prevention", "NTD elimination", "water-related diseases",
     "vector control", "integrated NTD control", "surveillance and response"],

    # 18. Non-Communicable Diseases and Mental Health
    ["non-communicable diseases", "NCDs", "cardiovascular disease", "diabetes", "chronic respiratory diseases",
     "cancer", "mental health", "risk factors", "NCD prevention", "tobacco control", "alcohol abuse",
     "physical activity", "healthy diets", "health promotion", "integrated services",
     "mental health integration",
     "depression", "anxiety disorders", "psychological first aid", "trauma-informed care"]
]


# ---------------------------------------------------------------------------
# Prompt template for OpenAI-based topic labeling
# ---------------------------------------------------------------------------

LABEL_PROMPT = """<|system|> You are an expert in topic modeling and global health policy, trained to create concise and accurate topic labels. Your goal is to generate the **most informative and specific** topic label possible based on the provided documents and keywords. </s>

<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Please generate a **concise topic label** (5 words or fewer) that best represents the theme of this topic. The label should be **specific, professional, and relevant to global health and international development**.  

If the topic is **unclear** based on the documents and keywords, return a **general but accurate** category instead (e.g., "Health Policy Trends" or "Global Health Financing").  

**Only return the topic label—no explanations, extra text, or formatting.** </s>  
<|assistant|>"""


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_topic_model() -> BERTopic:
    """
    Construct a fully configured BERTopic model ready for fitting.

    All hyperparameters come from ``config.py`` so they can be tuned without
    touching this module.
    """
    if not cfg.OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)

    umap_model = UMAP(
        n_neighbors=cfg.UMAP_N_NEIGHBORS,
        n_components=cfg.UMAP_N_COMPONENTS,
        min_dist=cfg.UMAP_MIN_DIST,
        spread=cfg.UMAP_SPREAD,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )

    cluster_model = KMeans(n_clusters=cfg.KMEANS_N_CLUSTERS)

    vectorizer_model = CountVectorizer(
        stop_words=get_enhanced_stopwords(),
        ngram_range=cfg.VECTORIZER_NGRAM_RANGE,
        min_df=cfg.VECTORIZER_MIN_DF,
        max_df=cfg.VECTORIZER_MAX_DF,
    )

    representation_model = OpenAIRepresentation(
        client, model=cfg.OPENAI_MODEL,
        prompt=LABEL_PROMPT, delay_in_seconds=cfg.OPENAI_DELAY, chat=True,
    )

    topic_model = BERTopic(
        verbose=True,
        representation_model=representation_model,
        nr_topics=cfg.NR_TOPICS,
        min_topic_size=cfg.MIN_TOPIC_SIZE,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=False),
        embedding_model=SentenceTransformer(cfg.EMBEDDING_MODEL_NAME),
        umap_model=umap_model,
        seed_topic_list=SEED_TOPICS,
        calculate_probabilities=True,
        language="english",
    )

    logger.info("BERTopic model built with %d seed topics", len(SEED_TOPICS))
    return topic_model


def train_and_save(
    documents: List[str],
    timestamps: List[int],
    administrations: List[str],
    output_dir: str,
) -> Tuple[BERTopic, list, list]:
    """
    Build, fit, visualize, and persist a BERTopic model.

    Returns (topic_model, topics, probs).
    """
    os.makedirs(output_dir, exist_ok=True)
    topic_model = build_topic_model()

    logger.info("Fitting BERTopic model on %d documents ...", len(documents))
    topics, probs = topic_model.fit_transform(documents)

    # ----- Save core artifacts -----
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)

    # Topics over time
    topics_over_time = topic_model.topics_over_time(
        documents, timestamps, nr_bins=cfg.TOPICS_OVER_TIME_BINS,
    )
    topics_over_time.to_csv(os.path.join(output_dir, "topics_over_time.csv"), index=False)

    # ----- Standard visualizations -----
    topic_model.visualize_topics_over_time(
        topics_over_time, top_n_topics=10, width=1200, height=800,
    ).write_html(os.path.join(output_dir, "topics_over_time.html"))

    topic_model.visualize_topics(
        width=1200, height=800,
    ).write_html(os.path.join(output_dir, "topic_network.html"))

    topic_model.visualize_hierarchy(
        width=1200, height=800,
    ).write_html(os.path.join(output_dir, "topic_hierarchy.html"))

    topic_model.visualize_barchart(
        top_n_topics=15, n_words=10, width=1200, height=800,
    ).write_html(os.path.join(output_dir, "topic_barchart.html"))

    # ----- Persist model (remove API client to avoid pickle issues) -----
    topic_model.representation_model = None
    model_path = os.path.join(output_dir, "topic_model")
    topic_model.save(model_path)
    logger.info("Model saved to %s", model_path)

    return topic_model, topics, probs
