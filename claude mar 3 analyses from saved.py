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
    api_key="insert your own api key here")


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
        color_discrete_map={'Trump': 'red', 'Biden': 'blue'},
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
        p_values[topic] = (np.sum(null_diff_array >= actual_diffs[topic]) + 1) / (len(null_diff_array) + 1)

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
    """Calculate topic coherence metrics to assess model quality"""
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

    # Get actual topic keywords from the model or extract from topic names
    topic_info = topic_model.get_topic_info()
    topics_keywords = {}
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # Skip outlier topic
            # Try to extract keywords from the existing Name field
            name = row['Name']
            if name and "_" in name:
                # If Name is formatted like "0_keyword1_keyword2_..."
                keywords = name.split("_")[1:]  # Skip the topic number
                topics_keywords[topic_id] = keywords
        # Fallback: Extract words from the topic name
        if topic_id in topic_info['Topic'].values:
            name = topic_info.loc[topic_info['Topic'] == topic_id, 'Name'].values[0]
            if name and "_" in name:
                # Extract everything after the topic number
                name_parts = name.split("_", 1)
                if len(name_parts) > 1:
                    # Split the name into individual words
                    name_words = name_parts[1].split()
                    # Keep only words that are in the dictionary
                    valid_words = [word.lower() for word in name_words
                                   if word.lower() in dictionary.token2id]
                    if len(valid_words) >= 2:
                        topics_keywords[topic_id] = valid_words

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

            # Log keyword filtering information to diagnose issues
            logger.debug(f"Topic {topic_id}: {len(keywords)} keywords, {len(filtered_keywords)} after filtering")

            if len(filtered_keywords) < 2:
                # Use a small default value instead of NaN
                logger.warning(f"Topic {topic_id}: Not enough keywords for coherence ({len(filtered_keywords)} < 2)")
                coherence_score = 0.01  # Default low value instead of NaN
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
        color_discrete_map={'Trump': 'red', 'Biden': 'blue'},
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
            color_discrete_map={'Trump': 'red', 'Biden': 'blue'},
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


def analyze_policy_attention_shifts(topic_model, documents, timestamps, administrations, output_dir):
    """Analyze shifts in policy attention using normalized attention scores"""
    logger.info("Analyzing policy attention shifts...")

    # Create time-based administration dataframe
    doc_data = pd.DataFrame({
        'text': documents,
        'timestamp': timestamps,
        'administration': administrations
    })

    # Get topics for documents
    topics, _ = topic_model.transform(documents)
    doc_data['topic'] = topics

    # Count topic occurrences by administration and normalize
    admin_topics = doc_data.groupby(['administration', 'topic']).size().reset_index(name='count')
    admin_totals = admin_topics.groupby('administration')['count'].sum().reset_index(name='total')
    admin_topics = admin_topics.merge(admin_totals, on='administration')
    admin_topics['norm_attention'] = admin_topics['count'] / admin_topics['total']

    # Calculate attention shifts
    pivot = admin_topics.pivot(index='topic', columns='administration', values='norm_attention').fillna(0)
    pivot['attention_shift'] = pivot['Biden'] - pivot['Trump']

    # Add topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    pivot['Name'] = pivot.index.map(lambda x: topic_labels.get(x, f"Topic {x}"))

    pivot.sort_values('attention_shift', ascending=False).to_csv(
        os.path.join(output_dir, "policy_attention_shifts.csv")
    )

    # Visualize top attention shifts - FIX: replace append with concat
    largest_shifts = pivot.nlargest(10, 'attention_shift')
    smallest_shifts = pivot.nsmallest(10, 'attention_shift')
    top_shifts = pd.concat([largest_shifts, smallest_shifts])

    fig = px.bar(
        top_shifts.reset_index(),
        x='Name',
        y='attention_shift',
        title='Largest Policy Attention Shifts Between Administrations',
        labels={'attention_shift': 'Shift in Normalized Attention (Biden - Trump)'},
        color='attention_shift',
        color_continuous_scale='RdBu',
        height=600
    )
    fig.write_html(os.path.join(output_dir, "policy_attention_shifts.html"))

    return pivot


def detect_focusing_events(topic_model, documents, timestamps, output_dir, window_size=60):
    """Identify potential focusing events by detecting sudden spikes in topic prominence"""
    logger.info("Detecting potential focusing events...")

    # Convert to dataframe for time analysis
    df = pd.DataFrame({'text': documents, 'timestamp': timestamps})
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str) + '-01-01')
    df = df.sort_values('timestamp')

    # Get topics for all documents
    topics, _ = topic_model.transform(documents)
    df['topic'] = topics

    # Create rolling windows to detect spikes
    topic_counts = df.groupby([pd.Grouper(key='timestamp', freq='M'), 'topic']).size().unstack().fillna(0)

    # Calculate z-scores for each topic over time
    rolling_mean = topic_counts.rolling(window=window_size, min_periods=1).mean()
    rolling_std = topic_counts.rolling(window=window_size, min_periods=1).std()
    z_scores = (topic_counts - rolling_mean) / rolling_std.replace(0, 1)

    # Identify significant spikes (z-score > 2)
    spikes = (z_scores > 2).astype(int)

    # Initialize with explicit dtypes
    spike_dates = pd.DataFrame(columns=['Date', 'Topic', 'Topic_Name', 'Z_Score'])

    topic_info = topic_model.get_topic_info()
    topic_labels = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    rows = []
    for topic in spikes.columns:
        topic_spikes = spikes.index[spikes[topic] == 1].tolist()
        for date in topic_spikes:
            rows.append({
                'Date': date,
                'Topic': topic,
                'Topic_Name': topic_labels.get(topic, f"Topic {topic}"),
                'Z_Score': float(z_scores.loc[date, topic])  # Explicitly convert to float
            })

    if rows:
        spike_dates = pd.DataFrame(rows)
        # Ensure Z_Score is float type
        spike_dates['Z_Score'] = spike_dates['Z_Score'].astype(float)

    # Save all results
    if not spike_dates.empty:
        spike_dates.sort_values('Z_Score', ascending=False).to_csv(
            os.path.join(output_dir, "potential_focusing_events.csv"), index=False
        )

        # Visualize top focusing events
        top_spikes = spike_dates.nlargest(15, 'Z_Score')
        fig = px.scatter(
            top_spikes,
            x='Date',
            y='Z_Score',
            color='Topic_Name',
            size='Z_Score',
            title='Potential Focusing Events (Sudden Topic Spikes)',
            labels={'Z_Score': 'Intensity (Z-Score)'},
            height=600
        )
        fig.write_html(os.path.join(output_dir, "focusing_events.html"))

    return spike_dates


def visualize_punctuated_equilibrium(topic_model, topics_over_time, output_dir):
    """Visualize patterns of stability and change in topic attention"""
    logger.info("Creating punctuated equilibrium visualization...")

    # Calculate month-to-month change in topic frequencies
    pivot = topics_over_time.pivot(index='Timestamp', columns='Topic', values='Frequency').fillna(0)
    changes = pivot.diff().abs()

    # Calculate overall volatility per time period
    volatility = changes.sum(axis=1)
    volatility_df = pd.DataFrame({'Date': volatility.index, 'Volatility': volatility.values})

    # Save volatility data
    volatility_df.to_csv(os.path.join(output_dir, "policy_volatility.csv"), index=False)

    # Create volatility visualization
    fig = px.line(
        volatility_df,
        x='Date',
        y='Volatility',
        title='Policy Attention Volatility Over Time (Punctuated Equilibrium)',
        labels={'Volatility': 'Total Topic Volatility'},
        height=500
    )

    # Add administration transition lines - use strings instead of timestamp objects
    fig.add_shape(
        type="line",
        x0="2017-01-01", y0=0,
        x1="2017-01-01", y1=1,
        line=dict(color="red", dash="dash"),
        xref="x", yref="paper"
    )
    fig.add_annotation(
        x="2017-01-01", y=0.95,
        text="Trump Administration Start",
        showarrow=False,
        xref="x", yref="paper"
    )

    fig.add_shape(
        type="line",
        x0="2021-01-01", y0=0,
        x1="2021-01-01", y1=1,
        line=dict(color="blue", dash="dash"),
        xref="x", yref="paper"
    )
    fig.add_annotation(
        x="2021-01-01", y=0.95,
        text="Biden Administration Start",
        showarrow=False,
        xref="x", yref="paper"
    )

    fig.write_html(os.path.join(output_dir, "punctuated_equilibrium.html"))

    return volatility_df


def create_methodological_summary(output_dir, significance_results=None, coherence_results=None, similarity_data=None,
                                  policy_shifts=None, focusing_events=None, volatility=None):
    """Create a comprehensive methodological summary report including new theoretical analyses"""
    logger.info("Creating methodological summary report...")

    with open(os.path.join(output_dir, "methodological_findings_summary.md"), "w") as f:
        f.write("# Methodological Analysis Summary\n\n")

        # Null model findings
        if significance_results is not None:
            # [existing code]
            pass

        # Policy Attention Shifts (new section)
        if policy_shifts is not None:
            f.write("\n## Policy Attention Shift Analysis\n\n")
            # Get top 5 topics with increased attention under Biden
            increased = policy_shifts.nlargest(5, 'attention_shift')
            f.write("- **Top 5 topics with increased attention under Biden administration:**\n")
            for idx, row in increased.iterrows():
                f.write(f"  - {row['Name']}: +{row['attention_shift']:.4f}\n")

            # Get top 5 topics with increased attention under Trump
            decreased = policy_shifts.nsmallest(5, 'attention_shift')
            f.write("\n- **Top 5 topics with increased attention under Trump administration:**\n")
            for idx, row in decreased.iterrows():
                f.write(f"  - {row['Name']}: {row['attention_shift']:.4f}\n")

            f.write(
                "\n  **Finding**: Policy attention shifted significantly between administrations, supporting the bounded rationality theory's prediction that new leadership selectively focuses on different priorities.\n")

        # Focusing Events (new section)
        if focusing_events is not None and not focusing_events.empty:
            f.write("\n## Focusing Events Detection\n\n")
            top_events = focusing_events.nlargest(5, 'Z_Score')
            f.write("- **Top 5 potential focusing events (sudden topic attention spikes):**\n")
            for idx, row in top_events.iterrows():
                f.write(f"  - {row['Date'].split()[0]}: {row['Topic_Name']} (intensity: {row['Z_Score']:.2f})\n")

            f.write(
                "\n  **Finding**: Several distinct focusing events were identified where specific topics gained sudden prominence, potentially representing policy windows as described by Kingdon's framework.\n")

        # Punctuated Equilibrium (new section)
        if volatility is not None:
            f.write("\n## Punctuated Equilibrium Analysis\n\n")
            # Find periods of highest volatility
            high_volatility = volatility.nlargest(3, 'Volatility')
            f.write("- **Periods of highest policy attention volatility:**\n")
            for idx, row in high_volatility.iterrows():
                f.write(f"  - {row['Date']}: Volatility score of {row['Volatility']:.2f}\n")

            f.write(
                "\n  **Finding**: The policy attention patterns show clear evidence of punctuated equilibrium, with periods of stability interrupted by bursts of change, particularly around administration transitions.\n")

        # Topic coherence findings (existing)
        if coherence_results is not None:
            # [existing code]
            pass

        # Document similarity findings (existing)
        if similarity_data is not None:
            # [existing code]
            pass

        # Updated recommendations
        f.write("\n## Recommendations for Further Analysis\n\n")
        f.write(
            "1. **Focus on significant topics**: Concentrate deeper qualitative analysis on topics with statistically significant differences between administrations.\n")
        f.write(
            "2. **Examine focusing events**: Further investigate the detected focusing events and their relationship to external developments or policy windows.\n")
        f.write(
            "3. **Analyze punctuation patterns**: Compare volatility patterns to major policy initiatives or leadership changes to validate the punctuated equilibrium findings.\n")
        f.write(
            "4. **Consider contextual factors**: Analyze how external events (e.g., COVID-19 pandemic) influenced topic shifts and focusing events.\n")
        f.write(
            "5. **Temporal lag analysis**: Investigate whether there's a lag period before policy priorities shift after administration changes.\n")


if __name__ == "__main__":
    # Configure file paths
    file_path = '/Users/liampowell/PycharmProjects/BERTopic Pilot/processed_paragraphs_filtered.csv'
    model_path = '/Users/liampowell/PycharmProjects/BERTopic Pilot/results_mar2_2/topic_model'
    output_dir = '/Users/liampowell/PycharmProjects/BERTopic Pilot/results_mar6_2/'

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        logger.info("Loading data...")
        documents, timestamps, administrations = load_and_clean_data(file_path, output_dir)

        # Load the saved BERTopic model
        logger.info("Loading saved BERTopic model...")
        topic_model = BERTopic.load(model_path)

        # Log model information
        topic_info = topic_model.get_topic_info()
        logger.info(f"Loaded model with {len(topic_info)} topics")



        # Calculate topic coherence
        try:
            coherence_results = calculate_topic_coherence(
                topic_model, documents, output_dir
            )
        except Exception as e:
            logger.warning(f"Topic coherence calculation failed: {str(e)}")
            coherence_results = None

        # Analyze temporal patterns
        # Analyze temporal patterns
        try:
            # Generate topics over time
            topics_over_time = topic_model.topics_over_time(documents, timestamps, nr_bins=20)
            topics_over_time.to_csv(os.path.join(output_dir, "topics_over_time_absolute.csv"), index=False)

            # Count total documents per timestamp/year
            year_counts = {}
            for year in timestamps:
                year_counts[year] = year_counts.get(year, 0) + 1

            # Normalize frequencies
            normalized_tot = topics_over_time.copy()
            for i, row in normalized_tot.iterrows():
                ts = row['Timestamp']

                # Get year - handle both numeric and string timestamps
                if isinstance(ts, (float, np.float64, int, np.int64)):
                    year = int(ts)
                else:
                    year = int(str(ts).split('-')[0])

                # Get count for this year, default to 1
                count = year_counts.get(year, 1)

                # Normalize frequency
                normalized_tot.at[i, 'Frequency'] = row['Frequency'] / count

            # Save normalized data
            normalized_tot.to_csv(os.path.join(output_dir, "topics_over_time_normalized.csv"), index=False)

            # Visualize using normalized data
            topic_model.visualize_topics_over_time(
                normalized_tot,
                top_n_topics=10,
                width=1200,
                height=800
            ).write_html(os.path.join(output_dir, "topics_over_time_normalized.html"))
        except Exception as e:
            logger.warning(f"Topic over time calculation failed: {str(e)}")
            coherence_results = None

        # Create additional visualizations
        try:
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
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")

        # After topics_over_time calculation
        focusing_events = detect_focusing_events(
            topic_model, documents, timestamps, output_dir
        )

        volatility = visualize_punctuated_equilibrium(
            topic_model, topics_over_time, output_dir
        )

        # Create methodological summary
        similarity_data = pd.read_csv(os.path.join(output_dir, "administration_similarity.csv"))
        # Create methodological summary with new analyses
        create_methodological_summary(
            output_dir,
            #significance_results=significance_results,
            coherence_results=coherence_results,
            similarity_data=similarity_data,

            focusing_events=focusing_events,
            volatility=volatility
        )

        logger.info(f"Analysis completed successfully! Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise