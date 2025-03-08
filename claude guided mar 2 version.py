import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import hdbscan
from umap import UMAP
from nltk.corpus import stopwords
from tqdm import tqdm
import logging
import re
import os
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from bertopic.representation import TextGeneration
from bertopic.representation import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import openai
import torch
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - User: liampwl - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key="your openai key")


def get_enhanced_stopwords() -> list:
    """Get comprehensive stopword list with fewer domain-specific removals"""
    # Standard English stopwords
    basic_stopwords = list(stopwords.words('english'))

    # Document/Publication specific terms - KEEP THESE
    publication_terms = [
        "prepared", "publication", "suggested", "citation", "isbn", "report",
        "washington", "dc", "agreement", "review", "official",
        "views", "positions", "contents", "executive", "summary",
        "background", "overview", "methodology", "appendix", "annex"
    ]

    # Document structure terms - KEEP THESE
    structure_terms = [
        "page", "section", "chapter", "paragraph", "figure", "table",
        "reference", "bibliography", "footnote", "note", "volume", "https", "www"
    ]

    # Time-related terms - KEEP THESE
    time_terms = [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
        "nov", "dec", "month", "monthly", "quarter", "quarterly", "annual",
        "annually", "year", "years", "fiscal", "fy", '2016', '2017', '2018', '2019', '2020',
        '2021', '2022', '2023', 'fy17', 'fy18'
    ]

    # Formatting and non-English terms - KEEP THESE
    unwanted_terms = [
        "pennsylvania ave", 'de', 'la', 'les', 'en', 'des', 'et', 'le', 'de la', 'du', 'para',
        'dan', 'pa', 'za', 'ke', 'ali', 'tes', 'primer', 'wa', "ts", "ee", "tr", "ea", "ti", "te",
        "ts", "ie", "io", "ro", "sb", "fn", "aa", "uo", "ye", 'na na na', 'na', 'na na',
        'tbd', '90 90', '600', '200', '85', '90'
    ]

    # Technical administrative terms - REDUCED LIST
    admin_terms = [
        "total", "number", "percent", "frequency", "details",
        "questionnaire", "respondents", "procedure", "guidelines"
    ]

    # USAID terms - REDUCED LIST
    org_terms = [
        "outreach", "reference", "budget", "procurement",
        "cooperative", "agreement", "implemented"
    ]

    # Combine all terms into a list - REMOVED ADDITIONAL_FROM_CHATGPT4O COMPLETELY
    all_stopwords = (basic_stopwords + publication_terms + structure_terms +
                     time_terms + unwanted_terms + admin_terms + org_terms)

    return list(set(all_stopwords))  # Remove duplicates while ensuring list format


def load_and_clean_data(file_path: str, output_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """Load and clean the dataset with detailed logging"""
    logger.info("Loading data from CSV...")

    # Read CSV file
    df = pd.read_csv(
        file_path,
        dtype={'year': int},
        low_memory=False
    )

    logger.info(f"Loaded {len(df)} paragraphs")
    logger.info(f"Columns found: {df.columns.tolist()}")

    # Clean text data with progress tracking
    logger.info("Cleaning text data...")
    tqdm.pandas(desc="Processing paragraphs")
    df['text'] = df['text'].astype(str)

    patterns_to_remove = [
        r'ISBN-: ----',
        r'AID-OAA-A--',
        r'\s{2,}',  # Multiple spaces
        r'\.{3,}',  # Three or more periods
        r'\([^\)]*\)',  # Text in parentheses
        r'\d+\.\d+\.\d+',  # Numbered sections
        r'^\s*\d+\.',  # Leading numbers with periods
        r'Figure \d+',  # Figure references
        r'Table \d+',  # Table references
    ]

    def clean_text(text: str) -> str:
        """Clean the text by applying the regex patterns"""
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text)  # Replace matching patterns with a space
        return ' '.join(text.split())  # Normalize spaces by splitting and joining

    # Apply the clean_text function to each document
    df['text'] = df['text'].apply(clean_text)

    # Add administration labels
    df['administration'] = df['year'].apply(
        lambda x: 'Biden' if x >= 2021 else 'Trump' if x >= 2017 else 'Other'
    )

    # Log dataset statistics
    logger.info(f"Final dataset: {len(df)} paragraphs")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    logger.info("Paragraphs per administration:")
    for admin, count in df['administration'].value_counts().items():
        logger.info(f"  {admin}: {count} paragraphs")

    # Count number of paragraphs per year
    yearly_counts = df['year'].value_counts().reset_index()
    yearly_counts.columns = ['year', 'num_paragraphs']
    yearly_counts = yearly_counts.sort_values(by='year')

    # Save the yearly paragraph counts to CSV
    yearly_counts_path = os.path.join(output_dir, "yearly_paragraph_counts.csv")
    yearly_counts.to_csv(yearly_counts_path, index=False)
    logger.info(f"Saved yearly paragraph counts to {yearly_counts_path}")

    return df['text'].tolist(), df['year'].tolist(), df['administration'].tolist()


# ===== NEW ANALYSIS FUNCTIONS =====

def analyze_administration_differences(topic_model, documents, administrations, output_dir):
    """
    Analyze and visualize topic distribution differences between administrations
    """
    logger.info("Analyzing topic distribution differences between administrations...")

    # Get topics for all documents
    topics, _ = topic_model.transform(documents)

    # Create lists for each administration
    trump_docs = [doc for doc, admin in zip(documents, administrations) if admin == 'Trump']
    biden_docs = [doc for doc, admin in zip(documents, administrations) if admin == 'Biden']

    # Get topics for each administration
    trump_topics, _ = topic_model.transform(trump_docs)
    biden_topics, _ = topic_model.transform(biden_docs)

    # Calculate topic frequency by administration
    trump_topic_freq = pd.Series(trump_topics).value_counts(normalize=True)
    biden_topic_freq = pd.Series(biden_topics).value_counts(normalize=True)

    # Combine into a DataFrame for comparison
    topic_comparison = pd.DataFrame({
        'Trump': trump_topic_freq,
        'Biden': biden_topic_freq
    }).fillna(0)

    # Add topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    topic_comparison['Label'] = topic_comparison.index.map(lambda x: topic_labels.get(x, f"Topic {x}"))

    # Calculate absolute difference
    topic_comparison['Difference'] = abs(topic_comparison['Trump'] - topic_comparison['Biden'])
    topic_comparison = topic_comparison.sort_values('Difference', ascending=False)

    # Save results
    topic_comparison.to_csv(os.path.join(output_dir, "administration_topic_comparison.csv"))

    # Create visualization for top differences
    top_diff = topic_comparison.head(15).copy()

    # Create diverging bar chart
    fig = px.bar(
        top_diff,
        x='Label',
        y=['Trump', 'Biden'],
        barmode='group',
        labels={'value': 'Topic Prevalence', 'variable': 'Administration'},
        title='Top 15 Topics with Greatest Shift Between Trump and Biden Administrations',
        height=600,
        width=1000,
    )

    fig.write_html(os.path.join(output_dir, "administration_topic_shift.html"))

    logger.info(f"Saved administration comparison to {os.path.join(output_dir, 'administration_topic_comparison.csv')}")

    return topic_comparison


def run_null_model_comparison(topic_model, documents, administrations, output_dir, n_permutations=100):
    """
    Compare actual topic distributions across administrations against a null model
    where administration labels are randomly shuffled.
    """
    logger.info(f"Running null model comparison with {n_permutations} permutations...")

    # Get actual topic assignments
    topics, _ = topic_model.transform(documents)

    # Get actual topic distributions by administration
    trump_docs = [doc for doc, admin in zip(documents, administrations) if admin == 'Trump']
    biden_docs = [doc for doc, admin in zip(documents, administrations) if admin == 'Biden']

    trump_topics, _ = topic_model.transform(trump_docs)
    biden_topics, _ = topic_model.transform(biden_docs)

    actual_trump_dist = pd.Series(trump_topics).value_counts(normalize=True)
    actual_biden_dist = pd.Series(biden_topics).value_counts(normalize=True)

    # Calculate actual differences
    all_topics = sorted(set(list(actual_trump_dist.index) + list(actual_biden_dist.index)))

    actual_diffs = {}
    for topic in all_topics:
        trump_freq = actual_trump_dist.get(topic, 0)
        biden_freq = actual_biden_dist.get(topic, 0)
        actual_diffs[topic] = abs(trump_freq - biden_freq)

    # Run permutation test
    null_diffs = {topic: [] for topic in all_topics}

    for i in tqdm(range(n_permutations), desc="Running permutations"):
        # Shuffle administration labels
        shuffled_administrations = np.random.permutation(administrations)

        # Get shuffled distributions
        shuffle_trump_docs = [doc for doc, admin in zip(documents, shuffled_administrations) if admin == 'Trump']
        shuffle_biden_docs = [doc for doc, admin in zip(documents, shuffled_administrations) if admin == 'Biden']

        if shuffle_trump_docs and shuffle_biden_docs:  # Ensure neither list is empty
            shuffle_trump_topics, _ = topic_model.transform(shuffle_trump_docs)
            shuffle_biden_topics, _ = topic_model.transform(shuffle_biden_docs)

            shuffle_trump_dist = pd.Series(shuffle_trump_topics).value_counts(normalize=True)
            shuffle_biden_dist = pd.Series(shuffle_biden_topics).value_counts(normalize=True)

            # Calculate differences in this permutation
            for topic in all_topics:
                trump_freq = shuffle_trump_dist.get(topic, 0)
                biden_freq = shuffle_biden_dist.get(topic, 0)
                null_diffs[topic].append(abs(trump_freq - biden_freq))

    # Calculate p-values
    p_values = {}
    for topic in all_topics:
        null_diff_array = np.array(null_diffs[topic])
        p_values[topic] = np.mean(null_diff_array >= actual_diffs[topic])

    # Compile results
    results = []
    topic_info = topic_model.get_topic_info()
    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    for topic in all_topics:
        results.append({
            'Topic': topic,
            'Name': topic_names.get(topic, f"Topic {topic}"),
            'Trump_Frequency': actual_trump_dist.get(topic, 0),
            'Biden_Frequency': actual_biden_dist.get(topic, 0),
            'Abs_Difference': actual_diffs[topic],
            'p_value': p_values[topic],
            'Significant': p_values[topic] < 0.05
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    # Save results
    results_df.to_csv(os.path.join(output_dir, "null_model_comparison.csv"), index=False)

    # Create visualization of significant differences
    significant = results_df[results_df['Significant']]

    if not significant.empty:
        plt.figure(figsize=(12, 8))
        plt.bar(
            range(len(significant)),
            significant['Abs_Difference'],
            tick_label=[
                f"{row['Topic']}: {row['Name'][:30]}..." if len(row['Name']) > 30 else f"{row['Topic']}: {row['Name']}"
                for _, row in significant.iterrows()]
        )
        plt.title('Significant Topic Differences Between Administrations')
        plt.ylabel('Absolute Difference in Topic Frequency')
        plt.xlabel('Topics')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "significant_topic_differences.png"), dpi=300)
        plt.close()

    logger.info(f"Completed null model comparison. Found {significant.shape[0]} significant topic differences.")
    return results_df


def calculate_topic_coherence(topic_model, documents, output_dir):
    """
    Calculate topic coherence metrics to assess model quality
    """
    logger.info("Calculating topic coherence metrics...")

    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Preprocess documents
    stop_words = set(stopwords.words('english'))

    logger.info("Tokenizing documents for coherence calculation...")
    tokenized_docs = []
    for doc in tqdm(documents[:5000], desc="Tokenizing documents"):  # Limit to 5000 docs for speed
        tokens = word_tokenize(doc.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
        tokenized_docs.append(tokens)

    # Create dictionary
    dictionary = Dictionary(tokenized_docs)

    # Get topic keywords
    topics_keywords = {}
    for topic_id in range(0, len(topic_model.get_topics())):
        topic_terms = [term for term, _ in topic_model.get_topic(topic_id)]
        topics_keywords[topic_id] = topic_terms

    # Calculate coherence
    coherence_measures = ["c_v", "u_mass"]
    results = []

    for measure in coherence_measures:
        logger.info(f"Calculating {measure} coherence...")
        for topic_id, keywords in tqdm(topics_keywords.items(), desc=f"Calculating {measure}"):
            if not keywords:
                continue

            # Filter keywords to those in dictionary
            filtered_keywords = [word for word in keywords if word in dictionary.token2id]

            if len(filtered_keywords) < 2:
                coherence_score = np.nan
            else:
                try:
                    cm = CoherenceModel(
                        topics=[filtered_keywords],
                        texts=tokenized_docs,
                        dictionary=dictionary,
                        coherence=measure
                    )
                    coherence_score = cm.get_coherence()
                except Exception as e:
                    logger.warning(f"Error calculating coherence for topic {topic_id}: {str(e)}")
                    coherence_score = np.nan

            topic_name = topic_info.loc[topic_info['Topic'] == topic_id, 'Name'].values[0] if topic_id in topic_info[
                'Topic'].values else f"Topic {topic_id}"
            results.append({
                'Topic': topic_id,
                'Name': topic_name,
                'Coherence_Measure': measure,
                'Score': coherence_score,
                'Num_Keywords': len(filtered_keywords)
            })

    # Create DataFrame and calculate stats
    coherence_df = pd.DataFrame(results)
    coherence_df.to_csv(os.path.join(output_dir, "topic_coherence_metrics.csv"), index=False)

    # Calculate average coherence
    avg_coherence = coherence_df.groupby('Coherence_Measure')['Score'].agg(['mean', 'median', 'std']).reset_index()
    avg_coherence.to_csv(os.path.join(output_dir, "average_topic_coherence.csv"), index=False)

    logger.info(
        f"Topic coherence calculation complete. Average c_v: {avg_coherence.loc[avg_coherence['Coherence_Measure'] == 'c_v', 'mean'].values[0]:.4f}")
    return coherence_df


def analyze_temporal_granularity(topic_model, documents, timestamps, output_dir):
    """
    Analyze topics at different temporal granularities
    """
    logger.info("Analyzing topics at different temporal granularities...")

    # Convert timestamps to datetime objects (assuming Jan 1st of each year)
    dates = [datetime(year=int(ts), month=1, day=1) for ts in timestamps]

    # Define granularities
    granularities = ["quarterly", "yearly"]
    results = {}

    for granularity in granularities:
        logger.info(f"Processing {granularity} granularity...")

        # Set number of bins based on granularity
        if granularity == "quarterly":
            nr_bins = 4 * (max(timestamps) - min(timestamps) + 1)  # 4 quarters per year
        else:  # yearly
            nr_bins = max(timestamps) - min(timestamps) + 1  # number of years

        # Calculate topics over time
        topics_over_time = topic_model.topics_over_time(
            documents,
            dates,
            nr_bins=nr_bins,
            evolution_tuning=True,
            global_tuning=True
        )

        # Save results
        results[granularity] = topics_over_time
        topics_over_time.to_csv(
            os.path.join(output_dir, f"topics_over_time_{granularity}.csv"),
            index=False
        )

        # Create visualization
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=10,
            width=1200,
            height=800
        )
        fig.write_html(os.path.join(output_dir, f"topics_over_time_{granularity}.html"))

    # Create topic evolution heatmap
    logger.info("Creating topic evolution heatmap...")

    # Use yearly granularity for the heatmap
    tot = results["yearly"]
    topic_info = topic_model.get_topic_info()

    # Get topic names
    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    # Add topic names to the data
    tot['TopicName'] = tot['Topic'].map(lambda x: topic_names.get(x, f"Topic {x}"))

    # Get top 15 topics by total frequency
    top_topics = tot.groupby('Topic')['Frequency'].sum().nlargest(15).index.tolist()
    filtered_tot = tot[tot['Topic'].isin(top_topics)]

    # Pivot for heatmap
    heatmap_data = filtered_tot.pivot_table(
        index='TopicName',
        columns='Timestamp',
        values='Frequency',
        aggfunc='mean'
    ).fillna(0)

    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Time", y="Topic", color="Frequency"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        title="Topic Evolution Over Time",
        height=800,
        width=1200
    )

    fig.write_html(os.path.join(output_dir, "topic_evolution_heatmap.html"))
    logger.info(f"Temporal analysis complete. Results saved to {output_dir}")

    return results


def analyze_document_similarity(topic_model, documents, administrations, output_dir, sample_size=2000):
    """
    Analyze document similarity across administrations
    """
    logger.info("Analyzing document similarity across administrations...")

    # Sample documents if there are too many
    if len(documents) > sample_size:
        logger.info(f"Sampling {sample_size} documents for similarity analysis...")

        # Ensure balanced sampling
        trump_indices = [i for i, admin in enumerate(administrations) if admin == 'Trump']
        biden_indices = [i for i, admin in enumerate(administrations) if admin == 'Biden']

        # Determine sample size for each administration
        trump_sample_size = min(len(trump_indices), sample_size // 2)
        biden_sample_size = min(len(biden_indices), sample_size // 2)

        # Sample indices
        np.random.seed(42)
        sampled_trump = np.random.choice(trump_indices, trump_sample_size, replace=False)
        sampled_biden = np.random.choice(biden_indices, biden_sample_size, replace=False)

        # Combine indices
        sampled_indices = np.concatenate([sampled_trump, sampled_biden])

        # Subset documents and administrations
        sampled_docs = [documents[i] for i in sampled_indices]
        sampled_admins = [administrations[i] for i in sampled_indices]
    else:
        sampled_docs = documents
        sampled_admins = administrations
        sampled_indices = list(range(len(documents)))

    # Extract embeddings
    logger.info("Extracting document embeddings...")
    embeddings = topic_model._extract_embeddings(
        sampled_docs,
        method="document",
        verbose=True
    )

    # Calculate t-SNE projection
    logger.info("Calculating t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Create DataFrame
    embedding_df = pd.DataFrame({
        'x': tsne_embeddings[:, 0],
        'y': tsne_embeddings[:, 1],
        'Administration': sampled_admins,
        'Original_Index': sampled_indices
    })

    # Save embeddings
    embedding_df.to_csv(os.path.join(output_dir, "document_embeddings.csv"), index=False)

    # Create scatter plot
    fig = px.scatter(
        embedding_df,
        x='x',
        y='y',
        color='Administration',
        title='Document Embedding Space by Administration',
        opacity=0.7,
        width=1000,
        height=800
    )
    fig.write_html(os.path.join(output_dir, "document_embedding_scatter.html"))

    # Calculate cross-administration similarity
    logger.info("Calculating cross-administration similarity...")

    # Get embeddings by administration
    trump_embeddings = embeddings[np.array(sampled_admins) == 'Trump']
    biden_embeddings = embeddings[np.array(sampled_admins) == 'Biden']

    # Calculate average similarity within and across administrations
    if len(trump_embeddings) > 0 and len(biden_embeddings) > 0:
        # Within Trump
        trump_sim = cosine_similarity(trump_embeddings)
        np.fill_diagonal(trump_sim, 0)  # Remove self-similarity
        avg_trump_sim = np.mean(trump_sim)

        # Within Biden
        biden_sim = cosine_similarity(biden_embeddings)
        np.fill_diagonal(biden_sim, 0)  # Remove self-similarity
        avg_biden_sim = np.mean(biden_sim)

        # Between Trump and Biden
        cross_sim = cosine_similarity(trump_embeddings, biden_embeddings)
        avg_cross_sim = np.mean(cross_sim)

        # Create similarity summary
        similarity_summary = pd.DataFrame([{
            'Within_Trump': avg_trump_sim,
            'Within_Biden': avg_biden_sim,
            'Between_Administrations': avg_cross_sim,
            'Trump_Documents': len(trump_embeddings),
            'Biden_Documents': len(biden_embeddings)
        }])

        similarity_summary.to_csv(os.path.join(output_dir, "administration_similarity.csv"), index=False)

        # Create bar chart
        sim_data = pd.DataFrame({
            'Comparison': ['Within Trump', 'Within Biden', 'Between Administrations'],
            'Average Similarity': [avg_trump_sim, avg_biden_sim, avg_cross_sim]
        })

        fig = px.bar(
            sim_data,
            x='Comparison',
            y='Average Similarity',
            title='Average Document Similarity Within and Across Administrations',
            width=800,
            height=500
        )
        fig.write_html(os.path.join(output_dir, "administration_similarity_chart.html"))

        logger.info(
            f"Document similarity analysis complete. Within Trump: {avg_trump_sim:.4f}, Within Biden: {avg_biden_sim:.4f}, Between: {avg_cross_sim:.4f}")

    return embedding_df


def analyze_topic_evolution(topic_model, topics_over_time, output_dir):
    """
    Analyze how topics evolve over time with statistical measures
    """
    logger.info("Analyzing topic evolution patterns...")

    # Create pivot table for analysis
    pivot = topics_over_time.pivot(index='Timestamp', columns='Topic', values='Frequency')

    # Calculate period-to-period changes
    changes = pivot.diff()

    # Get topics with biggest positive/negative changes
    total_change = changes.sum()
    increasing = total_change.nlargest(10)
    decreasing = total_change.nsmallest(10)

    # Get topic info for labels
    topic_info = topic_model.get_topic_info()
    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    # Create DataFrames for increasing/decreasing topics
    increasing_df = pd.DataFrame({
        'Topic': increasing.index,
        'Change_Value': increasing.values,
        'Name': [topic_names.get(t, f"Topic {t}") for t in increasing.index]
    })

    decreasing_df = pd.DataFrame({
        'Topic': decreasing.index,
        'Change_Value': decreasing.values,
        'Name': [topic_names.get(t, f"Topic {t}") for t in decreasing.index]
    })

    # Save results
    increasing_df.to_csv(os.path.join(output_dir, "increasing_topics.csv"), index=False)
    decreasing_df.to_csv(os.path.join(output_dir, "decreasing_topics.csv"), index=False)

    # Create visualizations
    fig_inc = px.bar(
        increasing_df,
        x='Name',
        y='Change_Value',
        title='Topics with Greatest Increase Over Time',
        labels={'Change_Value': 'Total Change in Frequency', 'Name': 'Topic'},
        height=600,
        width=900
    )
    fig_inc.write_html(os.path.join(output_dir, "increasing_topics.html"))

    fig_dec = px.bar(
        decreasing_df,
        x='Name',
        y='Change_Value',
        title='Topics with Greatest Decrease Over Time',
        labels={'Change_Value': 'Total Change in Frequency', 'Name': 'Topic'},
        height=600,
        width=900
    )
    fig_dec.write_html(os.path.join(output_dir, "decreasing_topics.html"))

    logger.info(f"Topic evolution analysis complete. Results saved to {output_dir}")
    return increasing_df, decreasing_df


def create_methodological_summary(output_dir, significance_results=None, coherence_results=None, similarity_data=None):
    """
    Create a summary report of methodological findings
    """
    logger.info("Creating methodological summary report...")

    with open(os.path.join(output_dir, "methodological_findings_summary.md"), "w") as f:
        f.write("# Methodological Analysis Summary\n\n")

        # Null model findings
        if significance_results is not None:
            f.write("## Significance Testing Results\n\n")
            sig_count = significance_results['Significant'].sum()
            total_topics = len(significance_results)
            f.write(
                f"- **{sig_count}** out of **{total_topics}** topics show significant differences between administrations (p<0.05)\n")
            if sig_count > 0:
                top_sig = significance_results[significance_results['Significant']].sort_values('p_value').head(5)
                f.write("- Top 5 most significant topics:\n")
                for _, row in top_sig.iterrows():
                    f.write(f"  - Topic {row['Topic']} ({row['Name']}): p={row['p_value']:.4f}\n")

        # Topic coherence findings
        if coherence_results is not None:
            f.write("\n## Topic Coherence Metrics\n\n")
            avg_coherence = coherence_results.groupby('Coherence_Measure')['Score'].mean().reset_index()
            for _, row in avg_coherence.iterrows():
                f.write(f"- Average {row['Coherence_Measure']} coherence: {row['Score']:.4f}\n")

            # List top 5 most coherent topics
            c_v_scores = coherence_results[coherence_results['Coherence_Measure'] == 'c_v'].sort_values('Score',
                                                                                                        ascending=False)
            if not c_v_scores.empty:
                f.write("- Top 5 most coherent topics (c_v):\n")
                for _, row in c_v_scores.head(5).iterrows():
                    f.write(f"  - Topic {row['Topic']} ({row['Name']}): {row['Score']:.4f}\n")

        # Document similarity findings
        if similarity_data is not None:
            f.write("\n## Document Similarity Analysis\n\n")
            for _, row in similarity_data.iterrows():
                f.write(f"- Average similarity within Trump documents: {row['Within_Trump']:.4f}\n")
                f.write(f"- Average similarity within Biden documents: {row['Within_Biden']:.4f}\n")
                f.write(f"- Average similarity between administrations: {row['Between_Administrations']:.4f}\n")

                # Calculate similarity ratio
                within_avg = (row['Within_Trump'] + row['Within_Biden']) / 2
                between = row['Between_Administrations']
                ratio = within_avg / between if between > 0 else float('inf')
                f.write(f"- Ratio of within-administration to between-administration similarity: {ratio:.2f}\n")

                if ratio > 1.1:
                    f.write(
                        "  - **Finding**: Documents are more similar within administrations than between them, suggesting distinct writing or framing styles.\n")
                else:
                    f.write(
                        "  - **Finding**: No strong evidence of distinct writing or framing styles between administrations.\n")

        f.write("\n## Recommendations for Further Analysis\n\n")
        f.write(
            "1. **Focus on significant topics**: Concentrate deeper qualitative analysis on topics with statistically significant differences between administrations.\n")
        f.write(
            "2. **Examine writing style differences**: Consider sentiment analysis to complement the document similarity findings.\n")
        f.write(
            "3. **Consider contextual factors**: Analyze how external events (e.g., COVID-19 pandemic) might have influenced topic shifts.\n")
        f.write(
            "4. **Temporal lag analysis**: Investigate whether there's a lag period before policy priorities shift after administration changes.\n")


if __name__ == "__main__":
    # Configure file paths
    file_path = '/Users/liampowell/PycharmProjects/BERTopic Pilot/processed_paragraphs_filtered.csv'
    output_dir = '/Users/liampowell/PycharmProjects/BERTopic Pilot/results_mar2_2/'

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load and clean data
        documents, timestamps, administrations = load_and_clean_data(file_path, output_dir)

        # Configure UMAP
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.1,
            spread=1.5,
            metric='cosine',
            random_state=42,
            low_memory=False,
        )

        # Configure HDBSCAN
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=50,
            metric='euclidean',
            min_samples=5,
            prediction_data=True
        )

        cluster_model = KMeans(n_clusters=50)

        # Get enhanced stopwords
        stopwords_list = get_enhanced_stopwords()

        # Configure CountVectorizer with enhanced stopwords
        vectorizer_model = CountVectorizer(
            stop_words=stopwords_list,
            ngram_range=(1, 3),
            min_df=20,
            max_df=0.95
        )

        # Define seed topics
        seed_topics = [
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

        # Load the Zephyr model and tokenizer


        # Set up the pipeline for text generation using Zephyr


        # Define the prompt template
        prompt = """<|system|> You are an expert in topic modeling and global health policy, trained to create concise and accurate topic labels. Your goal is to generate the **most informative and specific** topic label possible based on the provided documents and keywords. </s>

<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Please generate a **concise topic label** (5 words or fewer) that best represents the theme of this topic. The label should be **specific, professional, and relevant to global health and international development**.  

If the topic is **unclear** based on the documents and keywords, return a **general but accurate** category instead (e.g., "Health Policy Trends" or "Global Health Financing").  

**Only return the topic label—no explanations, extra text, or formatting.** </s>  
<|assistant|>"""

        # Create the representation model with OpenAI
        representation_model = OpenAI(client, model="gpt-4o", prompt=prompt, delay_in_seconds=10, chat=True)

        # Initialize BERTopic
        logger.info("Initializing BERTopic model...")
        topic_model = BERTopic(
            verbose=True,
            representation_model=representation_model,
            nr_topics="auto",
            min_topic_size=50,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=False),
            embedding_model=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
            umap_model=umap_model,
            seed_topic_list=seed_topics,
            calculate_probabilities=True,
            language="english"
        )

        # Fit the model
        logger.info("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(documents)

        # Save topic info
        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)

        # Standard visualizations and results
        logger.info("Creating standard visualizations and saving results...")

        # Topics over time
        topics_over_time = topic_model.topics_over_time(documents, timestamps, nr_bins=20)
        topics_over_time.to_csv(os.path.join(output_dir, "topics_over_time.csv"), index=False)

        topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=10,
            width=1200,
            height=800
        ).write_html(os.path.join(output_dir, "topics_over_time.html"))

        # Topic network
        topic_model.visualize_topics(
            width=1200,
            height=800
        ).write_html(os.path.join(output_dir, "topic_network.html"))

        # Topic hierarchy
        topic_model.visualize_hierarchy(
            width=1200,
            height=800
        ).write_html(os.path.join(output_dir, "topic_hierarchy.html"))


        # Topic barchart
        topic_model.visualize_barchart(
            top_n_topics=15,
            n_words=10,
            width=1200,
            height=800
        ).write_html(os.path.join(output_dir, "topic_barchart.html"))

        # NEW ENHANCED ANALYSES

        # 1. Administration differences analysis
        logger.info("Running enhanced analyses...")
        admin_topic_comparison = analyze_administration_differences(
            topic_model, documents, administrations, output_dir
        )

        # 2. Null model comparison (if time permits - can be computationally expensive)
        try:
            significance_results = run_null_model_comparison(
                topic_model, documents, administrations,
                output_dir, n_permutations=50  # Reduced from 100 for faster execution
            )
        except Exception as e:
            logger.warning(f"Null model comparison failed: {str(e)}")
            significance_results = None

        # 3. Topic coherence calculation
        try:
            coherence_results = calculate_topic_coherence(
                topic_model, documents, output_dir
            )
        except Exception as e:
            logger.warning(f"Topic coherence calculation failed: {str(e)}")
            coherence_results = None

        # 4. Temporal granularity analysis
        try:
            temporal_results = analyze_temporal_granularity(
                topic_model, documents, timestamps, output_dir
            )

            # 5. Topic evolution analysis
            increasing_topics, decreasing_topics = analyze_topic_evolution(
                topic_model, topics_over_time, output_dir
            )
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {str(e)}")
            temporal_results = None

        # 6. Document similarity analysis
        try:
            embedding_df = analyze_document_similarity(
                topic_model, documents, administrations,
                output_dir, sample_size=2000
            )
            similarity_data = pd.read_csv(os.path.join(output_dir, "administration_similarity.csv"))
        except Exception as e:
            logger.warning(f"Document similarity analysis failed: {str(e)}")
            similarity_data = None

        # Create methodological summary report
        create_methodological_summary(
            output_dir,
            significance_results=significance_results,
            coherence_results=coherence_results,
            similarity_data=similarity_data
        )

        # Save model
        logger.info("Saving model...")
        topic_model.representation_model = None  # Remove to avoid API client pickling issues
        topic_model.save(os.path.join(output_dir, "topic_model"))

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise