# EvoRMD: Integrating Biological Context and Evolutionary RNA Language Models for Interpretable Prediction of RNA Modifications

**EvoRMD** is a deep learning framework for multi-class prediction of RNA modifications from primary RNA sequence windows, augmented with multi-scale biological context.
The model combines:
RNA-FM (a large pretrained RNA language model)
A hierarchical anatomical encoder (species → organ/tissue → cell line → subcellular localization)
An adaptive attention pooling module and MLP classifier
to achieve state-of-the-art performance across 11 RNA modification types while providing cell- and state-specific motif interpretations.

---

## 🚀 Key Features

- **Supported RNA modification types (11)**
  - **Am**, **Cm**, **Um**, **Gm**, **D**, **Y**, **m¹A**, **m⁵C**, **m⁵U**, **m⁶A**, **m⁷G**
- **Evolutionary-aware sequence embeddings**
  - 41-nt window centered at the candidate site (20 nt upstream + 20 nt downstream)
  - Fed into **RNA-FM**, using the **12th (final) transformer layer** hidden states as contextual token embeddings
- **Biological-context encoder**
  - Encodes **species, organ/tissue, cell line, subcellular localization** into dense vectors
  - Fuses these context embeddings with RNA-FM token embeddings at every position to form a **biological content–aware representation**
- **Adaptive attention pooling**
  - Learns position-wise attention weights over the 41-nt window
  - Produces a **fused site-level embedding** via attention-weighted pooling
  - Enables **motif extraction** and **attention-based interpretability**
- **Interpretability utilities**
  - Extraction of **cell- and state-specific motifs**
  - Comparison of motifs across cell lines (e.g., **HepG2 vs Huh7**, **HNPCs vs GSCs**)
  - Visualization of **attention maps** and **latent-space separation** between modification types
- **Multi-label outputs per site**
  - For each candidate site, EvoRMD produces a **score / probability for all 11 RNA modification types simultaneously**.
  - By applying user-defined thresholds to these scores, a site can be assigned **zero、one or multiple** modification labels, enabling true **multi-label prediction**.
  
---

## 🧩 Prerequisites

- **Python**: 3.10.18
- **CUDA**: 12.8
- **PyTorch**: 2.3.1

> **Note:** Please ensure your CUDA + PyTorch versions are compatible with your GPU driver.
EvoRMD has been tested on NVIDIA RTX 3090 GPUs with 24 GB memory.

---

## 🛠 Installation

1. **Clone this repository**

    ```bash
    git clone https://github.com/Gardeina/EvoRMD.git
    cd EvoRMD
    ```

2. **Create and activate a environment**

    ```bash
    conda env create -f environment.yml
    conda activate evormd
    ```

---

## 📂 Usage

### 1️⃣ Data Preparation

To reproduce the main results in the paper, you can directly use the dataset provided in the GitHub Release:

- `RNAdata/all_data.csv`

Make sure it is placed under the `RNAdata/` directory:

```text
RNAdata/
  └── all_data.csv

This file already contains all samples required to train EvoRMD, including:

- **41-nt sequence windows** centered on candidate modification sites  
- **Modification labels** for 11 RNA modification types  
- **Biological context metadata:**
  - `species`
  - `organ` / `tissue`
  - `cell` / `cell_line`
  - `subcellular_location`


---


### 2️⃣ Model Training

Use PyTorch's distributed launcher to train:

    ```bash
    torchrun --nproc_per_node=<num_gpus> main.py \
        --raw_csv ./RNAdata/11_modif_preprocessed_data.pkl \
        --alpha 0.6 (default) \
        --downsamplingseed 42 (default) \
        --o ./Model/EVoRMD.pth \
        --result ./results.pkl \
        --num_epochs 100 (default) \
        --seed 42 (default) \
        --trainable 1 (default)
    ```


## ⚙️ Arguments

### 🔧 Key Arguments

- `--raw_csv`  
  Path to the **original raw dataset CSV file** (before downsampling).  
  Example: `RNAdata/all_data.csv`

- `--alpha`  
  Log-compression coefficient for **class-imbalance downsampling** (default: `0.6`).  
  Larger values apply stronger compression to very abundant classes.

- `--target_mods`  
  List of **modification types to downsample** (default: `["m6a", "m5c"]`).  
  Passed as a space-separated list, e.g.:  
  `--target_mods m6a m5c`

- `--min_keep`  
  Minimum number of samples to keep **per species** after downsampling (default: `1`).

- `--downsamplingseed`  
  Random seed used **only for the downsampling step** (default: `42`),  
  ensuring reproducible sub-sampling of abundant classes.

- `--mlp_depth`  
  Depth of the **MLP classifier head** (default: `1`).  
  - `1` = Linear classifier  
  - `2` = Two-layer MLP  
  - `3` = Three-layer MLP (best-performing setting in the paper)

- `--o`  
  Path to save the **trained model checkpoint** (`.pth` file).  
  Example: `Model/evormd_mlp3_trainable.pth`

- `--result`  
  Path to save the **training/validation/test results** (`.pkl` file),  
  including logits, probabilities, and attention weights.  
  Example: `Model/evormd_results.pkl`

- `--seed`  
  Global random seed for **reproducibility** (default: `42`).

- `--num_epochs`  
  Number of **training epochs** (default: `100`).

- `--trainable`  
  Whether to **fine-tune RNA-FM** or keep it frozen (default: `1`).  
  - `0` = freeze RNA-FM (feature extractor only)  
  - `1` = trainable RNA-FM (jointly optimized with MLP head)

---

## 🧪 Example Output

After training, you will see logs showing metrics such as:

- Overall Accuracy, Precision, Recall, F1-Score, MCC
- Per-class Accuracy and metrics

Example:

```
Class m6A:
  Accuracy: 0.9854
  Precision: 0.9921
  Recall: 0.9783
  F1-score: 0.9851
  MCC: 0.9835
```
---

## 📝 Citation

If you use **EvoRMD** in your research, please cite:

> 

---

## 📬 Contact

For questions or issues, please open an issue on GitHub or contact:

- Bo Wang: 2300393032@email.szu.edu.cn
