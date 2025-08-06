import os
import hashlib
import pandas as pd
import numpy as np
import PyPDF2
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2_contingency

from tensorflow import keras
from sentence_transformers import SentenceTransformer

import umap
import hdbscan

from fairlearn.metrics import demographic_parity_difference

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

import pickle
import logging
import seaborn as sns
from datetime import datetime
import google.generativeai as genai
import matplotlib.cm as cm

# --- 1. Initializations & Configuration ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
for sub_dir in ["plots", "autoencoder", "embeddings"]:
    os.makedirs(os.path.join(CACHE_DIR, sub_dir), exist_ok=True)

PLOT_CACHE_DIR = os.path.join(CACHE_DIR, "plots")
AUTOENCODER_CACHE_DIR = os.path.join(CACHE_DIR, "autoencoder")
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")

try:
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SBERT model loaded.")
except Exception as e:
    logger.error(f"Failed to load SBERT model: {e}")
    SBERT_MODEL = None

# --- 2. Data Ingestion & Utility Functions ---

def read_file(file):
    if file is None:
        raise ValueError("No file provided.")
    _, ext = os.path.splitext(file.name)
    ext = ext.lower()
    if ext == ".csv":
        return pd.read_csv(file.name)
    if ext == ".xlsx":
        return pd.read_excel(file.name)
    if ext == ".pdf":
        text = "".join([p.extract_text() for p in PyPDF2.PdfReader(file.name).pages if p.extract_text()])
        return pd.DataFrame({"text": text.split("\n")})
    raise ValueError(f"Unsupported file format: {ext}")

def hash_data(data):
    if isinstance(data, pd.DataFrame):
        return hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()
    return hashlib.md5(str(data).encode()).hexdigest()

def detect_data_type(df):
    # Detect a strong text column
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].dropna().astype(str).str.len().mean() > 30:
            return 'text', col
    return 'structured', None

# --- 3. Preprocessing and Embedding ---

def preprocess_structured(df, scaler_method="StandardScaler"):
    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=['object', 'category']).columns:
        df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))
    scaler = MinMaxScaler() if scaler_method == "MinMaxScaler" else StandardScaler()
    return scaler.fit_transform(df_proc)

def embed_text_with_caching(df, text_col):
    if SBERT_MODEL is None:
        raise RuntimeError("SentenceTransformer model is not available.")
    text_series = df[text_col].astype(str).fillna('')
    cache_file = os.path.join(EMBEDDINGS_CACHE_DIR, f"{hash_data(pd.DataFrame({'text': text_series}))}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    embeddings = SBERT_MODEL.encode(text_series.tolist(), show_progress_bar=True)
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

# --- 4. Advanced Clustering Functions ---

def autoencoder_clustering(X, hash_id, n_clusters, encoding_dim=32, epochs=100):
    cache_file = os.path.join(AUTOENCODER_CACHE_DIR, f"{hash_id}_{n_clusters}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    input_dim = X.shape[1]
    input_layer = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = keras.layers.Dense(64, activation='relu')(encoded)
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = keras.layers.Dense(64, activation='relu')(encoded)
    decoded = keras.layers.Dense(128, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = keras.Model(input_layer, decoded)
    encoder = keras.Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X, X, epochs=epochs, batch_size=256, verbose=0, callbacks=[early_stopping])
    X_enc = encoder.predict(X)
    labels = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit_predict(X_enc)
    with open(cache_file, "wb") as f:
        pickle.dump(labels, f)
    return labels

def run_clustering(X, method, n_clusters, dbscan_eps):
    logger.info(f"Running {method} clustering...")
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif method == 'dbscan':
        model = DBSCAN(eps=dbscan_eps, min_samples=5)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    if method == 'gmm':
        return model.fit(X).predict(X)
    return model.fit_predict(X)

def run_text_sbert_kmeans(df, text_col, n_clusters):
    X = embed_text_with_caching(df, text_col)
    return KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit_predict(X)

# --- 5. Metrics, Plotting, and Profiling ---

def plot_k_selection_graphs(X):
    max_k = min(15, len(X) - 1)
    if max_k < 2: return None, None
    k_values = range(2, max_k + 1)
    inertias, silhouette_scores = [], []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    suggested_k = k_values[np.argmax(silhouette_scores)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_title('Elbow Method'); ax1.grid(True)
    ax2.plot(k_values, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Score'); ax2.grid(True)
    ax2.axvline(x=suggested_k, color='g', linestyle='--', label=f'Suggested k = {suggested_k}'); ax2.legend()
    plot_path = os.path.join(PLOT_CACHE_DIR, 'k_selection_plot.png'); plt.savefig(plot_path); plt.close()
    return plot_path, suggested_k

def generate_cluster_profiles(df_original, cluster_labels):
    df_original['Discovered Cluster'] = cluster_labels
    profiles = {}
    for cluster_id in sorted(df_original['Discovered Cluster'].unique()):
        if cluster_id == -1: continue
        cluster_df = df_original[df_original['Discovered Cluster'] == cluster_id]
        profile = {"Cluster Size": f"{len(cluster_df)} ({len(cluster_df)/len(df_original):.1%})"}
        numeric_cols = cluster_df.select_dtypes(include=np.number).columns.drop('Discovered Cluster', errors='ignore')
        if not numeric_cols.empty: profile['Numeric Averages'] = cluster_df[numeric_cols].mean().round(2).to_dict()
        categorical_cols = cluster_df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            cat_profile = {col: dict(cluster_df[col].value_counts(normalize=True).nlargest(2).apply(lambda x: f"{x:.0%}")) for col in categorical_cols}
            profile['Top Categories'] = cat_profile
        profiles[f"Cluster {cluster_id}"] = profile
    return profiles

def save_plot(plot_func, filename, *args, **kwargs):
    plt.figure(figsize=kwargs.pop('figsize', (10, 8)))
    plot_func(*args, **kwargs)
    plt.tight_layout()
    filepath = os.path.join(PLOT_CACHE_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_cluster_profiles_heatmap(df, labels):
    df = df.copy()
    df['Discovered Cluster'] = labels
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None
    profile_means = numeric_df.groupby('Discovered Cluster').mean()
    scaler = MinMaxScaler()
    profile_scaled = scaler.fit_transform(profile_means)
    profile_scaled_df = pd.DataFrame(profile_scaled, index=profile_means.index, columns=profile_means.columns)
    plt.figure(figsize=(12, max(6, len(profile_scaled_df) * 0.7)))
    sns.heatmap(profile_scaled_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Normalized Mean Values for Numeric Features by Cluster', fontsize=16)
    plt.ylabel('Discovered Cluster')
    plt.tight_layout()
    filename = os.path.join(PLOT_CACHE_DIR, 'cluster_profile_heatmap.png')
    plt.savefig(filename)
    plt.close()
    return filename

def plot_silhouette(X, labels):
    from sklearn.metrics import silhouette_samples, silhouette_score
    from matplotlib import cm
    n_clusters = len(set(labels))
    if n_clusters < 2:
        return None
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    y_lower = 10
    for i in sorted(list(set(labels))):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("Silhouette Plot for the Various Clusters")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.tight_layout()
    filename = os.path.join(PLOT_CACHE_DIR, 'silhouette_plot.png')
    plt.savefig(filename)
    plt.close()
    return filename

def compute_silhouette_score(X, labels):
    if len(set(labels)) < 2:
        return "N/A"
    return round(silhouette_score(X, labels), 4)

def compute_unsupervised_fairness(df, sensitive_feature_col):
    y_pred = df['Discovered Cluster']
    sensitive_features = df[sensitive_feature_col]
    results = {}
    try:
        results["Demographic Parity Difference"] = round(demographic_parity_difference(y_pred, y_pred, sensitive_features=sensitive_features), 4)
    except:
        results["Demographic Parity Difference"] = "N/A"
    try:
        _, p, _, _ = chi2_contingency(pd.crosstab(y_pred, sensitive_features))
        results["Chi-Squared p-value"] = f"{p:.4f}"
    except:
        results["Chi-Squared p-value"] = "N/A"
    return results

# --- 6. Gemini & PDF Section ---

def get_pdf_styles():
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'Times-Bold'; styles['Title'].fontSize = 28; styles['Title'].alignment = TA_CENTER
    styles['h1'].fontName = 'Times-Bold'; styles['h1'].fontSize = 18; styles['h1'].textColor = colors.HexColor("#0b2545")
    styles['h2'].fontName = 'Times-Bold'; styles['h2'].fontSize = 14; styles['h2'].textColor = colors.HexColor("#0b2545")
    styles['Normal'].fontName = 'Times-Roman'; styles['Normal'].fontSize = 11; styles['Normal'].leading = 14
    styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER, textColor=colors.darkgray, spaceAfter=15))
    styles.add(ParagraphStyle(name='Footer', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.grey))
    return styles

def create_styled_table(data, colWidths):
    # Get standard style for text wrapping
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    # Wrap every cell as a Paragraph for wrapping & formatting
    wrapped_data = [
        [Paragraph(str(cell).replace(',', ',<br/>') if isinstance(cell, dict) else str(cell), normal_style)
         for cell in row]
        for row in data
    ]
    tbl = Table(wrapped_data, colWidths=colWidths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0b2545")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8)
    ]))
    return tbl


def build_executive_summary_prompt(report_data):
    # Build a professionally worded, robust prompt for Gemini

    # Get fairness metrics summary block
    fairness_lines = []
    fairness = report_data.get("Fairness Results", {})
    if isinstance(fairness, dict):
        for attr, metrics in fairness.items():
            fairness_lines.append(f"Sensitive Attribute: {attr}")
            if isinstance(metrics, dict):
                for metric, val in metrics.items():
                    fairness_lines.append(f"  - {metric}: {val}")
            else:
                fairness_lines.append(f"  - Value: {metrics}")
    elif fairness:
        fairness_lines.append(str(fairness))

    profiles = report_data.get("Cluster Profiles", {})
    profiles_text = "\n".join([f"- {name}: {str(profile)}" for name, profile in profiles.items()]) if isinstance(profiles, dict) else str(profiles)

    primary = report_data.get("Fairness Attribute", "N/A")
    method = report_data.get('Clustering Method', 'N/A')
    n_clusters = report_data.get('Number of Clusters', 'N/A')
    silhouette = report_data.get('Silhouette Score', 'N/A')

    prompt = f"""
As an expert data analyst, write a concise, three-paragraph executive summary for executive management using only the analytical results provided.

**Paragraph 1:** Begin by stating the goal of the analysis and enumerate the sensitive attribute(s) under review (e.g. {primary}). Clearly mention the clustering method employed ({method}), the number of clusters determined ({n_clusters}), and the calculated silhouette score ({silhouette}) indicating cluster quality.

**Paragraph 2:** Present the findings and insights. Interpret the fairness and bias metrics below in clear language, such as whether significant differences or associations were detected in the data (e.g. specific values of parity, p-values, or other key statistics). Summarize notable trends and groups found, referencing the cluster profiles as evidence for real-world groupings or disparities.

**Paragraph 3:** Summarize the implications of these findings for the business or organization. What does the data suggest about the composition and fairness of the groups discovered? What specific, actionable business or policy impacts or considerations arise strictly from the patterns found in this analysis? Do NOT advise on algorithmic or methodological matters—focus entirely on what the data and results mean for stakeholders.

Fairness, Disparity & Validation Results:
{chr(10).join(fairness_lines)}

Cluster Profiles:
{profiles_text}

**Remember:** Do not discuss how to improve analysis or methodology. Only narrate what the results and detected patterns mean for management. Structure your answer as three paragraphs, separated by blank lines, with professional language and clear flow.
"""
    return prompt


def query_gemini(report_data):
    import os
    api_key = os.getenv("GEMINI_API_KEY") or report_data.get("api_key")
    if not api_key:
        logger.error("GEMINI_API_KEY is not set in environment (cannot generate executive summary)")
        return " The Gemini API key is missing or invalid. Cannot generate executive summary."

    prompt = build_executive_summary_prompt(report_data)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        raw = model.generate_content(prompt)
        summary = raw.text.strip() if hasattr(raw, "text") else str(raw)
        return summary
    except Exception as e:
        error_str = str(e)
        logger.error(f"Gemini API call failed: {error_str}")
        # Handle quota/rate-limit errors gracefully
        if "429" in error_str or "quota" in error_str.lower():
            return (" Daily Gemini API quota reached: "
                    "No executive summary can be generated right now. "
                    "Please try again after your quota resets (usually at midnight UTC), or consider upgrading your Gemini API tier. "
                    "See: https://ai.google.dev/gemini-api/docs/rate-limits")
        return " An error occurred generating the summary. Please try again later."

def make_detailed_pdf_report(report_data, plot_paths):
    doc = SimpleDocTemplate(os.path.join(PLOT_CACHE_DIR, "Bias_Report.pdf"), pagesize=letter)
    styles = get_pdf_styles()
    story = [Spacer(1, 1.5*inch), Paragraph("The BiasNet", styles["Title"]), Paragraph("Developed by Anmol & Sudhanshu", styles['Subtitle']), PageBreak()]
    story.append(Paragraph("Executive Summary", styles["h1"]))
    summary = query_gemini(report_data)
    for para in summary.split('\n\n'):
        cleaned = para.strip()
        if cleaned:
            story.append(Paragraph(cleaned, styles['Normal']))
            story.append(Spacer(1, 0.15 * inch))
    story.append(PageBreak())



    story.append(Paragraph("Cluster Profiles", styles["h1"]))
    for name, profile in report_data.get("Cluster Profiles", {}).items():
        formatted_rows = []
        for k, v in profile.items():
            formatted_value = v
            if isinstance(v, dict):
                # Pretty-print dict nicely
                formatted_value = "<br/>".join(
                    [f"{kk}: {vv}" for kk, vv in v.items()]
                )
            formatted_rows.append([k, formatted_value])
        story.extend([
            Paragraph(name, styles["h2"]),
            create_styled_table(formatted_rows, colWidths=[2*inch, 5*inch]),
            Spacer(1, 0.2*inch)
        ])
    story.append(PageBreak())
    param_data = [['Clustering Method:', report_data.get('Clustering Method', 'N/A')],
                  ['Number of Clusters:', report_data.get('Number of Clusters', 'N/A')],
                  ['Silhouette Score:', f"{report_data.get('Silhouette Score', 'N/A')}"]]
    story.extend([Paragraph("Analysis Configuration", styles["h1"]),
                  create_styled_table(param_data, colWidths=[2.5*inch, 4.5*inch]), PageBreak()])
    if plot_paths.get('umap_cluster'):
        story.extend([Paragraph("Cluster Visualization (UMAP)", styles["h1"]),
                      Image(plot_paths['umap_cluster'], width=6*inch, height=4.5*inch), PageBreak()])
    story.append(Paragraph("Fairness Analysis", styles["h1"]))
    story.extend([Paragraph(f"Attribute: {report_data.get('Fairness Attribute', 'N/A').title()}", styles["h2"])])
    metric_data = [['Metric', 'Value', 'Interpretation'],
                   ["Demographic Parity", report_data.get("Fairness Results", {}).get("Demographic Parity Difference", "N/A"), "Closer to 0 is fairer."],
                   ["Chi-Squared p-value", report_data.get("Fairness Results", {}).get("Chi-Squared p-value", "N/A"), "< 0.05 suggests bias."]]
    story.append(create_styled_table(metric_data, colWidths=[2*inch, 1.5*inch, 3.5*inch]))
    if plot_paths.get('disparity'):
        story.extend([Spacer(1, 0.3*inch), Image(plot_paths['disparity'], width=7*inch, height=4*inch)])
    story.append(Paragraph("End of Report", styles["Footer"]))
    doc.build(story)
    return doc.filename

# --- 7. Main Pipeline ---

def bias_discovery_pipeline(file, p_attr, s_attr, removed_cols, scaler, method, k, eps, progress):
    progress(0, desc="Starting...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY secret not found.")
    df = read_file(file)
    df_original = df.copy()
    if removed_cols:
        df.drop(columns=removed_cols, inplace=True, errors='ignore')
    sensitive_cols_to_drop = [p_attr]
    if s_attr and s_attr != "None":
        fairness_attr = f"{p_attr}_{s_attr}"
        df[fairness_attr] = df[p_attr].astype(str) + "_" + df[s_attr].astype(str)
        df_original[fairness_attr] = df[fairness_attr]
        sensitive_cols_to_drop.append(s_attr)
    else:
        fairness_attr = p_attr
    progress(0.2, desc="Preprocessing...")
    data_type, text_col = detect_data_type(df)
    df_for_clustering = df.drop(columns=sensitive_cols_to_drop, errors='ignore')

    # CLUSTERING SELECTION (all methods supported)
    if method == 'autoencoder':
        X = preprocess_structured(df_for_clustering, scaler)
        labels = autoencoder_clustering(X, hash_data(df_for_clustering), k)
    elif method == 'sbert_kmeans':
        if data_type != 'text':
            raise ValueError("sbert_kmeans works for text columns only.")
        labels = run_text_sbert_kmeans(df_for_clustering, text_col, k)
        X = embed_text_with_caching(df_for_clustering, text_col)
    elif method == 'spectral':
        X = preprocess_structured(df_for_clustering, scaler)
        labels = run_clustering(X, method, k, eps)
    else:
        if data_type == 'structured':
            X = preprocess_structured(df_for_clustering, scaler)
            labels = run_clustering(X, method, k, eps)
        else:
            X = embed_text_with_caching(df_for_clustering, text_col)
            labels = run_clustering(X, 'kmeans', k, eps)
    df_original['Discovered Cluster'] = labels
    progress(0.5, desc="Profiling & Visualizing...")
    profiles = generate_cluster_profiles(df, labels)
    plot_paths = {}
    # Visualizations
    if df_original[fairness_attr].nunique() <= 20:
        plot_paths['disparity'] = save_plot(
            lambda: sns.countplot(data=df_original, x='Discovered Cluster', hue=fairness_attr, palette='viridis'),
            'disparity_plot.png'
        )
    plot_paths['umap_cluster'] = save_plot(
        lambda: sns.scatterplot(
            x=umap.UMAP(random_state=42).fit_transform(X)[:, 0],
            y=umap.UMAP(random_state=42).fit_transform(X)[:, 1],
            hue=labels, palette='Spectral', s=20
        ),
        'umap_plot_cluster.png'
    )
    plot_paths['profile_heatmap'] = plot_cluster_profiles_heatmap(df_original.copy(), labels)
    plot_paths['silhouette'] = plot_silhouette(X, labels)

    report_data = {
        "api_key": api_key,
        "Clustering Method": method,
        "Number of Clusters": len(set(labels)),
        "Silhouette Score": compute_silhouette_score(X, labels),
        "Fairness Attribute": fairness_attr,
        "Fairness Results": compute_unsupervised_fairness(df_original, fairness_attr),
        "Cluster Profiles": profiles
    }
    progress(0.9, desc="Generating PDF...")
    pdf_path = make_detailed_pdf_report(report_data, plot_paths)
    csv_path = "clustered_output.csv"
    df_original.to_csv(csv_path, index=False)
    progress(1, desc="Complete!")
    return df_original, X, plot_paths, profiles, report_data["Fairness Results"], pdf_path, csv_path

# --- 8. Gradio Interface ---

def gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as demo:
        gr.Markdown("# BiasNet: Unsupervised Bias Discovery Engine\n#### Developed by Anmol & Sudhanshu")
        df_original_state, X_state, plot_paths_state, text_mode_state = gr.State(), gr.State(), gr.State(), gr.State(value=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configuration")
                file_input = gr.File(label="Upload Dataset")
                p_sensitive_input = gr.Dropdown(label="Primary Sensitive Attribute")
                s_sensitive_input = gr.Dropdown(label="Secondary Sensitive Attribute (Optional)")
                removed_cols_input = gr.Dropdown(label="Columns to Remove (Optional)", multiselect=True)
                with gr.Accordion("Advanced Settings", open=False):
                    scaler_input = gr.Radio(["StandardScaler", "MinMaxScaler"], value="StandardScaler", label="Scaler")

                gr.Markdown("### 2. Analysis Settings")
                method_input = gr.Dropdown(
                    ['kmeans', 'agglomerative', 'gmm', 'dbscan', 'hdbscan', 'autoencoder', 'spectral'], # sbert_kmeans added dynamically
                    value='kmeans', label="Clustering Method"
                )
                cluster_input = gr.Number(value=4, label="Num. Clusters (k)", precision=0)
                dbscan_eps_slider = gr.Slider(0.1, 2.0, value=0.5, label="DBSCAN Epsilon", visible=False)
                k_button = gr.Button("Find & Suggest k")
                run_button = gr.Button("Analyze for Bias", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 3. Results")
                error_output = gr.Textbox(label="Status / Errors", interactive=False, visible=False)
                with gr.Tabs():
                    with gr.Tab("Fairness Analysis"):
                        fairness_output = gr.JSON(label="Fairness Metrics Summary")
                        plot_output_fairness = gr.Image(show_label=False)
                    with gr.Tab("Cluster Profiles"):
                        profile_output = gr.JSON(label="Profile of Each Discovered Cluster")
                    with gr.Tab("Cluster Deep Dive"):
                        gr.Markdown("#### Cluster Profile Heatmap\nNormalized mean values of numeric features for each cluster.")
                        plot_output_heatmap = gr.Image(show_label=False)
                        gr.Markdown("#### Silhouette Plot\nVisualizes cluster density and separation. Taller, wider bars are better.")
                        plot_output_silhouette = gr.Image(show_label=False)
                    with gr.Tab("Visualizations"):
                        umap_color_input = gr.Dropdown(label="Color UMAP Plot By", interactive=True)
                        plot_output_viz = gr.Image(show_label=False)
                    with gr.Tab("Model Diagnostics"):
                        plot_output_diag = gr.Image(label="Optimal k Plot", show_label=False)
                with gr.Row():
                    pdf_output = gr.File(label="Download Report (PDF)", visible=True)
                    csv_output = gr.File(label="Download Data (CSV)", visible=True)

        # File upload + dynamic method detection for text data
        def on_file_upload(file):
            if not file:
                return (gr.update(choices=[]),)*4 + (False, )
            try:
                df = read_file(file)
                cols = ["None"] + df.columns.tolist()
                cols_for_color = ["Discovered Cluster"] + df.columns.tolist()
                data_type, text_col = detect_data_type(df)
                text_mode = (data_type == 'text')
                return (gr.update(choices=cols[1:]), gr.update(choices=cols),
                        gr.update(choices=cols[1:]), gr.update(choices=cols_for_color, value="Discovered Cluster"), text_mode)
            except Exception as e:
                gr.Warning(f"Read error: {e}")
                return (gr.update(choices=[]),)*4 + (False, )

        file_input.upload(
            on_file_upload, file_input,
            [p_sensitive_input, s_sensitive_input, removed_cols_input, umap_color_input, text_mode_state]
        )

        # Dynamic enabling/disabling of sbert_kmeans
        def update_methods(text_mode):
            all_methods = ['kmeans', 'agglomerative', 'gmm', 'dbscan', 'hdbscan', 'autoencoder', 'spectral', 'sbert_kmeans']
            if text_mode:
                return gr.update(choices=all_methods, value='sbert_kmeans')
            else:
                return gr.update(choices=[m for m in all_methods if m != "sbert_kmeans"], value="kmeans")
        text_mode_state.change(update_methods, text_mode_state, method_input)

        def toggle_k_input(method):
            needs_k = method in ['kmeans', 'agglomerative', 'gmm', 'autoencoder', 'spectral', 'sbert_kmeans']
            is_dbscan = method == 'dbscan'
            return gr.update(visible=needs_k), gr.update(visible=is_dbscan)
        method_input.change(toggle_k_input, method_input, [cluster_input, dbscan_eps_slider])

        def run_preliminary_analysis(file, progress=gr.Progress(track_tqdm=True)):
            if not file:
                gr.Warning("Upload a file first.")
                return None, None, gr.update(visible=True, value="File not uploaded.")
            progress(0.5, desc="Finding optimal k...")
            try:
                df = read_file(file)
                data_type, text_col = detect_data_type(df)
                X = embed_text_with_caching(df, text_col) if data_type == 'text' else preprocess_structured(df)
                k_plot, suggested_k = plot_k_selection_graphs(X)
                if k_plot is None:
                    gr.Warning("Not enough data to find optimal k.")
                    return None, None, gr.update(visible=False)
                gr.Info(f"Analysis complete. Suggested k = {suggested_k}")
                return gr.update(value=suggested_k), k_plot, gr.update(visible=False)
            except Exception as e:
                gr.Warning(f"Error: {e}")
                return None, None, gr.update(value=str(e), visible=True)

        k_button.click(run_preliminary_analysis, file_input, [cluster_input, plot_output_diag, error_output])

        def run_main_analysis(file, p_attr, s_attr, removed, scaler, method, k, eps, progress=gr.Progress(track_tqdm=True)):
            try:
                if not file or not p_attr:
                    gr.Warning("Upload file & select Primary Sensitive Attribute.")
                    return [None]*11
                df_original, X, plots, profiles, fairness, pdf, csv = bias_discovery_pipeline(
                    file, p_attr, s_attr, removed, scaler, method, k, eps, progress
                )
                return (df_original, X, plots, profiles, fairness, pdf, csv,
                        gr.update(visible=True, value=plots.get('disparity')),
                        gr.update(visible=True, value=plots.get('umap_cluster')),
                        plots.get('profile_heatmap'), plots.get('silhouette'))
            except ValueError as e:
                logger.error(f"FATAL error in pipeline: {e}", exc_info=True)
                gr.Error(f"Error: {e}")
                return [None]*11
            except ImportError as e:
                logger.error(f"Keras/tensorflow problem: {e}")
                gr.Error("Tensorflow/Keras not installed. Please run `pip install tensorflow tf-keras` and restart.")
                return [None]*11
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                gr.Error(f"Error: {e}")
                return [None]*11

        run_button.click(
            run_main_analysis,
            inputs=[file_input, p_sensitive_input, s_sensitive_input, removed_cols_input,
                    scaler_input, method_input, cluster_input, dbscan_eps_slider],
            outputs=[df_original_state, X_state, plot_paths_state, profile_output, fairness_output,
                     pdf_output, csv_output, plot_output_fairness, plot_output_viz,
                     plot_output_heatmap, plot_output_silhouette]
        )

        def generate_colored_umap(df, X, color_col):
            if df is None or X is None or not color_col: return None
            labels = df[color_col] if color_col != "Discovered Cluster" else df['Discovered Cluster']
            return save_plot(lambda: sns.scatterplot(
                x=umap.UMAP(random_state=42).fit_transform(X)[:, 0],
                y=umap.UMAP(random_state=42).fit_transform(X)[:, 1],
                hue=labels, palette='viridis', s=20),
                f'umap_colored_{color_col}.png'
            )
        umap_color_input.change(
            generate_colored_umap, [df_original_state, X_state, umap_color_input], plot_output_viz
        )

        return demo

if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
