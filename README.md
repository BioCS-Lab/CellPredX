# CellPredX

CellPredX is a computational framework for **cross-modality**, **cross-sample**, and **cross-protocol** cell type annotation.  
It integrates **domain adaptation** and **deep metric learning** to transfer labels reliably between single-cell RNA sequencing (scRNA-seq), single-cell ATAC sequencing (scATAC-seq), and related multi-omic assays (e.g. scRNA-seq, scATAC-seq).

---

## 1. Overview

Accurate cell type annotation across heterogeneous single-cell datasets remains a major challenge due to:

- distinct experimental protocols and platforms,
- batch effects and sample-specific biases, and
- the presence of cell types that are absent in the reference data.

CellPredX addresses these issues by learning a **shared latent representation** for reference and query datasets, in which:

- cells with the same biological identity are aligned across modalities,
- dataset-specific and protocol-specific variations are reduced, and
- known and potentially novel cell types can be systematically evaluated.

The framework is designed to support both **closed-set** (all query cell types are known) and **open-set** (novel cell types present in the query) scenarios.

---

## 2. Key Features

- **Cross-modality label transfer**  
  Supports label transfer between scRNA-seq, scATAC-seq, and multi-omic data (e.g. CITE-seq and ASAP-seq).

- **Domain adaptation and metric learning**  
  Learns modality-invariant embeddings using a combination of projection regularization, contrastive learning, alignment losses, and center-based regularization.

- **Open-set recognition**  
  Evaluates the ability to detect novel cell types using AUROC and open-set classification rate (OSCR).

- **Support for multiple public datasets**  
  Includes example configurations for MCA, HFA, PBMC, and CITE-ASAP.

- **Reproducible end-to-end tutorial**  
  A comprehensive Jupyter notebook (`examples/Tutorial_Demo.ipynb`) demonstrates the full workflow from data preprocessing to model training, evaluation, and interpretation of outputs.

---

## 3. Installation

### 3.1 Clone the repository

```bash
git clone https://github.com/BioCS-Lab/CellPredX.git
cd CellPredX
```

### 3.2 Create a Python environment (recommended)

```bash
conda create -n cellpredx python=3.9 -y
conda activate cellpredx
```

### 3.3 Install dependencies

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

Otherwise, install the core dependencies manually, for example:

```bash
pip install \
  numpy pandas scipy scikit-learn \
  scanpy anndata \
  torch torchvision \
  matplotlib seaborn h5py
```

---
## 4. Data Availability

CellPredX has been tested on several public benchmark datasets. The following pairs are used in our examples:

| Dataset    | RNA Modality                                          | ATAC / Multi-omic                         |
|-----------|--------------------------------------------------------|-------------------------------------------|
| MCA       | [Tabula Muris RNA](https://tabula-muris.ds.czbiohub.org/) | [Mouse ATAC Atlas](https://atlas.gs.washington.edu/mouse-atac/) |
| HFA       | [GSE156793](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793) | [GSE149683](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683) |
| PBMC      | [Seurat RNA + ATAC tutorial data](https://satijalab.org/seurat/articles/atacseq_integration_vignette.html) | – |
| CITE-ASAP | [scJoint CITE-ASAP data](https://github.com/SydneyBioX/scJoint) | ASAP-seq (paired ADT + ATAC)              |

Users can either download these datasets following the original instructions or adapt CellPredX to their own datasets as long as the input is formatted as `AnnData` objects with appropriate feature and metadata annotations.


## 5. Comprehensive Tutorial (Recommended Entry Point)

We strongly recommend starting with the Jupyter notebook:

> **`examples/Tutorial_Demo.ipynb`**

This notebook provides a **step-by-step, fully reproducible workflow** including:

1. Loading scRNA-seq, scATAC-seq and ADT data as `AnnData` objects.  
2. Configuring preprocessing (binarization, log-transformation, scaling, highly variable feature selection, and optional ADT concatenation).  
3. Constructing ATAC-side raw embeddings and k-nearest-neighbor graphs for neighborhood-preserving losses.  
4. Instantiating the `BuildCellpredX` model with user-defined hyperparameters (latent dimensionality, loss weights, temperature, etc.).  
5. Training the model and monitoring multiple loss components:  
   - Projection regularization / NNDR-style terms  
   - Feature alignment loss  
   - Contrastive (InfoNCE) loss  
   - Sparse center loss  
   - Supervised cross-entropy loss on reference data  
6. Evaluating performance on:  
   - shared (known) cell types using accuracy and macro-F1, and  
   - novel (unknown) cell types using AUROC and OSCR.  
7. Interpreting the main output matrices, including latent embeddings, predicted probabilities, and inferred labels for query cells.

The notebook is intended to serve as a **comprehensive tutorial**, particularly for new users, and reflects the full workflow described in the manuscript.

---

## 6. Minimal Usage Example (CITE-ASAP)

The following example illustrates how to use CellPredX programmatically on the CITE-ASAP dataset. For clarity, only the essential steps are shown; see `examples/Tutorial_Demo.ipynb` for the complete pipeline.

```python
import os
from os.path import join
import scanpy as sc
import scipy.sparse as sps

from CellPredX import BuildCellpredX  # adjust according to your package structure

data_root = "/path/to/scNCL_data/"

# 1. Load RNA and ATAC data
adata_rna = sc.read_h5ad(join(data_root, "CITE-ASAP/adata_rna_cache.h5ad"))
adata_atac = sc.read_h5ad(join(data_root, "CITE-ASAP/adata_atac_cache.h5ad"))

# 2. Load ADT (CITE-seq and ASAP-seq)
adata_cite_adt = sc.AnnData(
    sps.load_npz(join(data_root, "CITE-ASAP/citeseq_control_adt.npz")),
    obs=adata_rna.obs,
)
adata_asap_adt = sc.AnnData(
    sps.load_npz(join(data_root, "CITE-ASAP/asapseq_control_adt.npz")),
    obs=adata_atac.obs,
)

# 3. ATAC-side raw embedding (e.g. ADT-based representation)
atac_raw_emb = adata_asap_adt.X.A

# 4. Preprocessing configuration
ppd = dict(
    binz=True,
    hvg_num=adata_atac.shape[1],
    lognorm=False,
    scale_per_batch=False,
    batch_label="domain",
    type_label="cell_type",
    knn=10,
    knn_by_tissue=False,
)

# 5. Instantiate the model
model = BuildCellpredX(
    n_latent=48,
    bn=False,
    dr=0.2,
    cont_w=0.2,
    cont_tau=0.8,
    mycenter_w=0.1,
    center_cutoff=0,
)

# 6. Preprocess and construct input matrices
atac_number_label, label_dict = model.preprocess(
    adata_inputs=[adata_rna, adata_atac],
    atac_raw_emb=atac_raw_emb,
    adata_adt_inputs=[adata_cite_adt, adata_asap_adt],
    pp_dict=ppd,
)

# 7. Train the model (example hyperparameters)
output_dir = "outputs/CITE-ASAP-demo"
os.makedirs(output_dir, exist_ok=True)

model.train(
    opt="sgd",
    batch_size=512,
    training_steps=1000,
    lr=0.01,
    lr2=None,
    weight_decay=5e-4,
    log_step=50,
    eval_atac=True,
    eval_top_k=2,
    eval_open=True,
    output_dir=output_dir,
)

# 8. Evaluate and obtain embeddings and predicted labels
feat_A, feat_B, head_A, head_B = model.eval(inplace=True)
atac_pred_labels = model.annotate(label_prop=False)
```

In typical analyses, the resulting latent embeddings (`feat_A`, `feat_B`) and predicted labels (`atac_pred_labels`) are used for downstream visualization (e.g. UMAP) and biological interpretation.

---

## 7. Method Summary

Conceptually, CellPredX optimizes a composite objective that includes:

- **Projection regularization (PR) loss**  
  Encourages the latent space to be well-conditioned, low-correlated, and informative by penalizing deviations from zero-centered, decorrelated, and sufficiently variable embeddings.

- **Feature alignment (FA) loss**  
  Aligns reference and query embeddings via cosine similarity, enhanced by negative sampling to discourage incorrect matches across modalities.

- **Cross-entropy (CE) loss**  
  Supervises the reference branch using known cell type labels, promoting discriminative, cell type-specific features.

- **Contrastive loss (CL)**  
  Preserves neighborhood structure in the query modality by pulling together cells that are close in the raw ATAC-derived embedding while pushing apart unrelated cells.

- **Sparse center loss (SCL)**  
  Encourages compact clusters in latent space for both true labels (reference) and high-confidence pseudo-labels (query), thereby sharpening decision boundaries while maintaining sparsity.

Together, these components enable robust and flexible label transfer across diverse single-cell datasets and experimental settings.

---

## 8. Repository Structure

A typical layout of the repository is:

```text
CellPredX/
  ├── core.py                   # Core implementation of BuildCellpredX
  ├── model.py                  # Encoder and classification head definitions
  ├── dataset.py                # Dataset and preprocessing utilities
  ├── loss.py                   # Loss functions (PR, CL, SCL, etc.)
  ├── metrics.py                # Evaluation metrics (accuracy, F1, OSCR, AUROC)
  ├── utils.py                  # Helper functions
  ├── examples/
  │   └── Tutorial_Demo.ipynb   # Comprehensive end-to-end tutorial (recommended)
  ├── requirements.txt
  └── README.md
```

(Exact file names may differ slightly in the actual implementation; the above serves as a conceptual guide.)

---

## 9. License

This project is released under the **MIT License**.

© 2025 BioCS-Lab. All rights reserved.

---

## 10. Contact

For questions, bug reports, or suggestions, please open an issue on GitHub:

- Issues: https://github.com/BioCS-Lab/CellPredX/issues  

For collaboration inquiries, please contact the corresponding authors as listed in the manuscript.

---

## 11. Citation

If you use CellPredX in your research, we kindly ask you to cite the corresponding paper:

> *Citation details will be added here once the manuscript is published.*
