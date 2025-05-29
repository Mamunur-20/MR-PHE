# MR-PHE: Multi-Resolution Prompt-guided Hybrid Embedding for Zero-Shot Histopathology Classification

This repository contains the official implementation of the paper: **"Leveraging Vision-Language Embeddings for Zero-Shot Learning in Histopathology Images"**.

*Authors: Md Mamunur Rahaman, Ewan K. A. Millar, and Erik Meijering, Fellow, IEEE*

## 📖 Abstract

Zero-shot learning (ZSL) holds tremendous potential for histopathology image analysis by enabling models to generalize to unseen classes without extensive labeled data. Recent advancements in vision-language models (VLMs) have expanded the capabilities of ZSL, allowing models to perform tasks without task-specific fine-tuning. However, applying VLMs to histopathology presents considerable challenges due to the complexity of histopathological imagery and the nuanced nature of diagnostic tasks. In this paper, we propose a novel framework called Multi-Resolution Prompt-guided Hybrid Embedding (MR-PHE) to address these challenges in zero-shot histopathology image classification. MR-PHE leverages multiresolution patch extraction to mimic the diagnostic workflow of pathologists, capturing both fine-grained cellular details and broader tissue structures critical for accurate diagnosis. We introduce a hybrid embedding strategy that integrates global image embeddings with weighted patch embeddings, effectively combining local and global contextual information. Additionally, we develop a comprehensive prompt generation and selection framework, enriching class descriptions with domain-specific synonyms and clinically relevant features to enhance semantic understanding. A similarity-based patch weighting mechanism assigns attention-like weights to patches based on their relevance to class embeddings, emphasizing diagnostically important regions during classification. Our approach utilizes pretrained VLM, CONCH for ZSL without requiring domain-specific fine-tuning, offering scalability and reducing dependence on large annotated datasets. Experimental results demonstrate that MR-PHE not only significantly improves zero-shot classification performance on histopathology datasets but also often surpasses fully supervised models, highlighting its superior effectiveness and potential for advancing computational pathology.

## ✨ Key Features

*   **Multi-Resolution Patch Extraction**: Mimics pathologist diagnostic workflow by analyzing images at varying scales to capture both fine-grained cellular details and broader tissue structures.
*   **Hybrid Embedding Strategy**: Integrates global image embeddings with weighted patch embeddings derived from multi-resolution patches, combining local and global contextual information
*   **Comprehensive Prompt Generation & Selection**: Generates domain-specific textual prompts enriched with synonyms and clinically relevant features, followed by a filtering process to select the most effective prompts.
*   **Similarity-based Patch Weighting**: Assigns attention-like weights to patches based on their relevance to class embeddings, emphasizing diagnostically important regions.
*   **Zero-Shot Learning (ZSL)**: Utilizes the pretrained CONCH Vision-Language Model for ZSL without requiring domain-specific fine-tuning, offering scalability.


## ⚙️ Requirements

*   Python 3.8+
*   PyTorch (refer to `requirements.txt` for version)
*   Torchvision (refer to `requirements.txt` for version)
*   NumPy
*   PyYAML
*   Tqdm
*   Pillow
*   Fire
*   scikit-learn
*   Matplotlib
*   Pandas
*   python-dotenv
*   (See `requirements.txt` for a complete list of dependencies and versions)

## 🛠️ Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/your_username/MR-PHE-Histopathology.git
    cd MR-PHE-Histopathology
    ```

2.  **Create a virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```
    pip install -r requirements.txt

    ```
    pip install -e .
    ```


## 📊 Datasets

This project supports multiple histopathology datasets. You will need to download them and organize them as expected by the custom data loaders in `src/my_datasets/`. Refer to `data/README.md`.

Supported datasets include (but are not limited to, based on loader scripts:
*   WSSS (e.g., class names: "Normal", "Stroma", "Tumor")
*   HEGHIDS
*   BRACS (and BRACS7Class)
*   EBHI
*   CRC (Colorectal Cancer)


## ⚙️ Configuration

*   Experiment parameters (e.g., model size, batch size, dataset specific settings, patch extraction parameters, hybrid weight, temperature) are defined in YAML files located in the `configs/` directory. Examples: `conch_wsss.yaml`, `conch_heghids.yaml` [6], `wca_ebhi.yaml` [4], `conch_wsss_evaluation.yaml` [1].
*   The main scripts (e.g., `src/main_scripts/mrphe_classify.py`) accept a `--config` argument to specify which configuration file to use.
*   Machine-specific paths (e.g., `DATA_ROOT_PATH`, `MODEL_CHECKPOINT_PATH`) and sensitive keys (e.g., `HF_AUTH_TOKEN`) should be set in the `.env` file. The `src/utils/config_loader.py` (you'll create this) should handle loading these.

## 🚀 Quick Start / Usage Examples

Ensure your datasets are set up and the `.env` file is configured before running the scripts.

### 1. Generate Prompts
This script generates textual prompts based on class names, synonyms, and clinical significance.



