# EvoRMD: Integrating Biological Context and Evolutionary RNA Language Models for Interpretable Multi-label Prediction of RNA Modifications

**EvoRMD** is a deep learning framework for **multi-label prediction** of RNA modifications from primary RNA sequence windows, augmented with multi-scale biological context.

The model combines:

- **RNA-FM** (a large pretrained RNA language model)
- A **hierarchical anatomical encoder** (species → organ/tissue → cell line → subcellular localization)
- An **adaptive attention pooling module** and **MLP classifier**

to achieve strong performance across 11 RNA modification types while providing **cell- and state-specific motif interpretations**.

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
  - By applying user-defined thresholds to these scores, a site can be assigned **zero, one, or multiple** modification labels, enabling true **multi-label prediction**.

---

## 🧩 Prerequisites

- **Python**: 3.10.18  
- **CUDA**: 12.8  
- **PyTorch**: 2.3.1  

> **Note:** Please ensure your CUDA + PyTorch versions are compatible with your GPU driver.  
> EvoRMD has been tested on NVIDIA RTX 3090 GPUs with 24 GB memory.

---

## 🛠 Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/Gardeina/EvoRMD.git
   cd EvoRMD
