import os
import h5py
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import csv
import gzip
import scipy.io
import scipy.sparse as sps
import matplotlib.pyplot as plt
from os.path import join
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize
from metrics import osr_evaluator
from core import BuildCellpredX
import utils as utls
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(1234)
sc.settings.verbosity = 3
sc.logging.print_header()
exp_id = 'CITE-ASAP' + '-1201'  # dataset_name + timestamp

data_root = '/data/yan_code/scNCL-release/data/scNCL_data/'

adata_rna = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_rna_cache.h5ad'))
adata_atac = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_atac_cache.h5ad'))

meta_rna = adata_rna.obs
meta_atac = adata_atac.obs

meta = pd.concat([meta_rna, meta_atac], axis=0)
# proteomics data, already lognormalized
adata_cite_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/citeseq_control_adt.npz')), obs=adata_rna.obs)
adata_asap_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/asapseq_control_adt.npz')), obs=adata_atac.obs)
atac_raw_emb = adata_asap_adt.X.A
# params dict of preprocessing 
ppd = {'binz': True, 
       'hvg_num':adata_atac.shape[1], 
       'lognorm':False, 
       'scale_per_batch':False,  
       'batch_label': 'domain',
       'type_label':  'cell_type',
       'knn': 10,
       'knn_by_tissue':False
       }  # default settings

#  outputs folder
output_dir = join(f'/data/yan_code/scDWL/outputs/{exp_id}')
os.makedirs(output_dir, exist_ok=True)

model = BuildCellpredX(
                n_latent=48, bn=False, dr=0.2, 
                cont_w=0.2, cont_tau=0.5,mycenter_w=0.005,center_cutoff=0,
        )
    
model.preprocess(
                [adata_rna, adata_atac],   # list of 'anndata' object
                atac_raw_emb,   
                adata_adt_inputs=[adata_cite_adt, adata_asap_adt], # 
                pp_dict = ppd
        )
if 1:
    model.train(
        batch_size=512, training_steps=5000, 
        lr=0.01, 
        log_step=50, eval_atac=True, output_dir=output_dir,#eval_top_k=1, eval_open=True,  
    )
else:
    # loading checkpoints
        ckpt_path = join(output_dir, 'ckpt_cite.pth')
        model.load_ckpt(ckpt_path)
##############################################
model.eval(inplace=True)
atac_pred_type= model.annotate()
ad_atac = sc.AnnData(model.feat_B)
ad_atac.obs = meta_atac.copy()
ad_atac.obs['pred_type'] = atac_pred_type
ad_atac.obs['pred_conf'] = np.max(model.head_B, axis=1)
ad_atac.obs['pred_type']
share_mask = meta_atac.cell_type.isin(meta_rna.cell_type.unique()).to_numpy()
open_score = 1 - np.max(model.head_B, axis=1) # pb_max, logit_max_B

kn_data_pr = atac_pred_type[share_mask]
kn_data_gt = meta_atac.cell_type[share_mask].to_numpy()
kn_data_open_score = open_score[share_mask]

unk_data_open_score = open_score[np.logical_not(share_mask)]

closed_acc, os_auroc, os_aupr, oscr = osr_evaluator(kn_data_pr, kn_data_gt, kn_data_open_score, unk_data_open_score)
#############################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(meta_atac.cell_type.to_numpy(), atac_pred_type)
cm = cm/cm.sum(axis=1, keepdims=True)

df_cm = pd.DataFrame(cm, index = meta_atac.cell_type.unique(),
                  columns = meta_atac.cell_type.unique())

plt.figure(figsize = (10,7))
sns.heatmap(df_cm, )