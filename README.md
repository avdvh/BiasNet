# BiasNet: Unsupervised Bias Discovery Engine


BiasNet is a comprehensive, end-to-end data analysis tool designed to uncover hidden biases and inequities in datasets without requiring pre-existing labels. It serves as an exploratory engine for data scientists, analysts, and decision-makers to audit their data for potential disparities across sensitive demographic attributes.

The application leverages a suite of unsupervised machine learning algorithms to segment a population into naturally occurring groups. By analyzing the demographic composition and characteristics of these discovered clusters, the tool quantifies potential biases. The entire workflow is encapsulated in an interactive web interface, and the final output is a detailed, multi-page PDF report that includes an AI-generated executive summary, detailed cluster profiles, fairness metrics, and rich visualizations.


**Try it live**: [Click Here](https://huggingface.co/spaces/avdvh/BiasNet)


---

## Table of Contents

* [Core Features](#core-features)
* [Project Structure](#project-structure)
* [Methodology Deep Dive](#methodology-deep-dive)
* [Technical Stack](#technical-stack)
* [Installation and Setup](#installation-and-setup)
* [How to Use](#how-to-use)
* [Generated Outputs](#generated-outputs)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Core Features

* **Versatile Data Ingestion**: Accepts data in multiple common formats, including CSV (`.csv`), Excel (`.xlsx`), and PDF (`.pdf`). For PDFs, it automatically extracts text and structures it for analysis.

* **Automatic Data Type Detection**: Intelligently determines if the dataset is primarily structured (tabular) or unstructured (text-based) to apply the appropriate analysis pipeline.

* **Advanced Preprocessing & Embedding**:
    * **Structured Data**: Handles categorical features via label encoding and numeric features with user-selectable scaling (`StandardScaler` or `MinMaxScaler`).
    * **Text Data**: Utilizes a powerful `Sentence-BERT` model (`all-MiniLM-L6-v2`) to generate meaningful numerical embeddings from text content.

* **Comprehensive Clustering Suite**: Implements a wide range of clustering algorithms to suit different data shapes and analysis needs:
    * **Centroid-based**: K-Means
    * **Hierarchical**: Agglomerative Clustering
    * **Distribution-based**: Gaussian Mixture Model (GMM)
    * **Density-based**: DBSCAN, HDBSCAN
    * **Graph-based**: Spectral Clustering
    * **Deep Learning-based**: Autoencoder for dimensionality reduction followed by K-Means.
    * **Text-specific**: SBERT Embeddings + K-Means.

* **Unsupervised Fairness Metrics**: Quantifies bias using established statistical methods:
    * **Demographic Parity Difference**: Measures if clusters are formed disproportionately across different subgroups of a sensitive attribute.
    * **Chi-Squared Test of Independence**: Assesses the statistical significance of the association between cluster assignments and sensitive attributes.

* **Rich Interactive Visualizations**:
    * **UMAP Cluster Visualization**: Provides a 2D representation of high-dimensional data for visual inspection. The plot can be interactively colored by any feature.
    * **Disparity Distribution Plot**: A bar chart showing the distribution of sensitive attribute subgroups within each discovered cluster.
    * **Cluster Profile Heatmap**: A normalized heatmap of numeric feature means for each cluster, allowing for easy comparison of cluster characteristics.
    * **Silhouette Plot**: Visualizes the density and separation of clusters, helping to validate the quality of the clustering result.

* **AI-Powered Reporting**:
    * **Gemini Executive Summary**: Queries the Google Gemini API with a structured prompt containing the analysis results to generate a concise, three-paragraph executive summary.
    * **Detailed PDF Report**: Compiles all outputs into a professional, multi-page PDF document generated using `reportlab`.

* **Efficient Caching**: Caches computationally expensive operations like SBERT embeddings and trained autoencoder models to disk, significantly speeding up subsequent runs on the same data.

---

## Project Structure

The repository is organized into a core application module and a user-facing interface script for clarity and maintainability.

```
BiasNet/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py                  # Main entry point to launch the Gradio web application
│
└── biasnet/                # Core source code module
    ├── __init__.py
    ├── analysis.py         # Functions for cluster profiling, fairness metrics, and visualizations
    ├── clustering.py       # Implementations of all clustering algorithms
    ├── data_processing.py  # Data ingestion, preprocessing, and embedding logic
    ├── reporting.py        # PDF generation and Gemini API integration
    └── utils.py            # Utility functions like caching and data hashing
```

---

## Methodology Deep Dive

The BiasNet pipeline is a sequence of modular steps, each handled by a specific component of the `biasnet` package.

1.  **Data Ingestion & Preprocessing (`biasnet/data_processing.py`)**: The pipeline begins by reading the user-uploaded file. This module detects the data type (structured vs. text) and applies the necessary preprocessing steps, such as `LabelEncoding` for categorical data or generating `Sentence-BERT` embeddings for text.

2.  **Clustering (`biasnet/clustering.py`)**: The preprocessed data is then passed to the selected clustering algorithm. This module contains wrappers for various `scikit-learn` models, as well as custom implementations for `autoencoder`-based clustering (using TensorFlow/Keras) and density-based methods like `HDBSCAN`.

3.  **Analysis & Visualization (`biasnet/analysis.py`)**: Once clusters are assigned, this module takes over. It generates detailed cluster profiles by calculating statistics for each group. It then computes fairness metrics using the `fairlearn` library and statistical tests from `scipy.stats`. Finally, it produces all visualizations using `matplotlib`, `seaborn`, and `UMAP`.

4.  **Reporting (`biasnet/reporting.py`)**: The final step involves compiling all results. This module constructs a detailed prompt with the quantitative results and queries the Google Gemini API for an executive summary. It then uses `reportlab` to assemble the summary, profiles, metrics, and plots into a polished, downloadable PDF report.

---

## Technical Stack

* **Backend & Data Analysis**: Python 3.9+
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`, `hdbscan`, `tensorflow`, `keras`, `fairlearn`
* **NLP & Embeddings**: `sentence-transformers`
* **Dimensionality Reduction**: `umap-learn`
* **Visualization**: `matplotlib`, `seaborn`
* **Web Interface**: `gradio`
* **PDF Generation**: `reportlab`
* **AI Summarization**: `google-generativeai`
* **File Handling**: `pypdf2`

---

## Installation and Setup

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/avdvh/BiasNet.git](https://github.com/avdvh/BiasNet.git)
    cd BiasNet
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**: Install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variable**: The application requires a Google Gemini API key to generate the executive summary.
    * Obtain a key from [Google AI Studio](https://ai.google.dev/).
    * Set it as an environment variable named `GEMINI_API_KEY`.

    On Linux/macOS:
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    On Windows (Command Prompt):
    ```bash
    set GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    Alternatively, you can create a `.env` file in the root directory and add the line `GEMINI_API_KEY="YOUR_API_KEY_HERE"`.

---

## How to Use

1.  **Launch the Application**: Run the `app.py` script from the root directory.
    ```bash
    python app.py
    ```
    This will launch the Gradio web server. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser.

2.  **Configure the Analysis**:
    * **Upload Dataset**: Upload your `.csv`, `.xlsx`, or `.pdf` file.
    * **Select Attributes**: Choose the Primary Sensitive Attribute and, optionally, a Secondary one. Select any columns to exclude from the analysis.
    * **Choose Method**: Select a clustering algorithm and configure its parameters (e.g., number of clusters `k`). Use the **"Find & Suggest k"** button for guidance.

3.  **Run and Review**:
    * Click the **"Analyze for Bias"** button to start the pipeline.
    * Explore the results in the output tabs: review fairness metrics, inspect cluster profiles, and interact with the visualizations.
    * Download the final **PDF Report** and the augmented **CSV Data** using the file links provided.

---

## Generated Outputs

1.  **PDF Report (`Bias_Report.pdf`)**: A comprehensive, professionally styled report containing:
    * An AI-generated **Executive Summary**.
    * Detailed **Cluster Profiles** with quantitative and qualitative descriptions.
    * **Analysis Configuration** parameters.
    * **Visualizations** (UMAP plot, Disparity plot, Heatmap).
    * A summary of **Fairness Metrics** and their interpretations.

2.  **Clustered Data (`clustered_output.csv`)**: A CSV file containing all the data from the original uploaded file, with a new `Discovered Cluster` column appended, which holds the cluster assignment for each row.

---

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them with descriptive messages.
4.  Push your changes to your forked repository.
5.  Create a new Pull Request, detailing the changes you have made.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

* This tool was developed by [**Anmol**](https://github.com/avdvh) & [**Sudhanshu**](https://github.com/xudhanxhu)
* Built with the amazing capabilities of the open-source Python data science ecosystem.
