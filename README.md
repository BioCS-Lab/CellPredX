# 🎯 CellPredX

> **CellPredX**: A computational framework for **cross-data type**, **cross-sample**, and **cross-protocol** cell type annotation through **domain adaptation** and **deep metric learning**.

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/BioCS-Lab/CellPredX.git
cd CellPredX
```

---

## 📂 Datasets

Supported public datasets:

| Dataset     | RNA Modality                                                                                   | ATAC / Multiomic                                                                 |
|-------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **MCA**     | [🔗 Tabula Muris RNA](https://tabula-muris.ds.czbiohub.org/)                                   | [🔗 Mouse ATAC Atlas](https://atlas.gs.washington.edu/mouse-atac/)              |
| **HFA**     | [🔗 GSE156793](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793)                   | [🔗 GSE149683](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683)    |
| **PBMC**    | [🔗 Seurat RNA+ATAC](https://satijalab.org/seurat/articles/atacseq_integration_vignette.html) | –                                                                                |
| **CITE-ASAP** | [🔗 scJoint](https://github.com/SydneyBioX/scJoint)                                          | –                                                                                |

---

## 🚀 Quick Start

An example using **CITE-ASAP** data:

```python
import scanpy as sc
import pandas as pd
import scipy.sparse as sps
import os
from os.path import join
from CellPredX import BuildCellpredX

# Load RNA & ATAC data
adata_rna = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_rna_cache.h5ad'))
adata_atac = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_atac_cache.h5ad'))

# Merge metadata
meta = pd.concat([adata_rna.obs, adata_atac.obs], axis=0)

# Load proteomics (log-normalized)
adata_cite_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/citeseq_control_adt.npz')), obs=adata_rna.obs)
adata_asap_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/asapseq_control_adt.npz')), obs=adata_atac.obs)

# ASAP ATAC embedding
atac_raw_emb = adata_asap_adt.X.A

# Preprocessing settings
ppd = {
    'binz': True,
    'hvg_num': adata_atac.shape[1],
    'lognorm': False,
    'scale_per_batch': False,
    'batch_label': 'domain',
    'type_label': 'cell_type',
    'knn': 10,
    'knn_by_tissue': False
}

# Create output directory
myoutput_dir = join(f'/data/yan_code/scDWL/outputs/{exp_id}')
os.makedirs(myoutput_dir, exist_ok=True)

# Initialize model
model = BuildCellpredX(
    n_latent=48, bn=False, dr=0.2,
    cont_w=0.2, cont_tau=0.8,
    mycenter_w=0.1, center_cutoff=0,
)

# Preprocess & predict
atac_number_label, label_dict = model.preprocess(
    [adata_rna, adata_atac],
    atac_raw_emb,
    adata_adt_inputs=[adata_cite_adt, adata_asap_adt],
    pp_dict=ppd
)
```

---

## 📄 License

This project is licensed under the MIT License.  
© 2025 [BioCS-Lab](https://github.com/BioCS-Lab)

---

## 📬 Contact

For questions or collaborations, feel free to open an [issue](https://github.com/BioCS-Lab/CellPredX/issues) or email the authors.

---

## 💡 Citation

> *Citation coming soon... Stay tuned!*
