data_root = '../data/scNCL_data'
/data/yan_code/scNCL-release/data/scNCL_data/CITE-ASAP/adata_atac_cache.h5ad
adata_rna = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_rna_cache.h5ad'))
adata_atac = sc.read_h5ad(join(data_root, 'CITE-ASAP/adata_atac_cache.h5ad'))

meta_rna = adata_rna.obs
meta_atac = adata_atac.obs

meta = pd.concat([meta_rna, meta_atac], axis=0)

adata_cite_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/citeseq_control_adt.npz')), obs=adata_rna.obs)
adata_asap_adt = sc.AnnData(sps.load_npz(join(data_root, 'CITE-ASAP/asapseq_control_adt.npz')), obs=adata_atac.obs)

adata_cite_adt, adata_asap_adt

# low-dimension representations of raw scATAC-seq data
atac_raw_emb = adata_asap_adt.X.A
atac_raw_emb.shape

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
output_dir = join(f'../outputs/{exp_id}')
os.makedirs(output_dir, exist_ok=True)


model = BuildscNCL(
                n_latent=64, bn=False, dr=0.2, 
                cont_w=0.2, cont_tau=0.4,
        )
    
model.preprocess(
                [adata_rna, adata_atac],   # list of 'anndata' object
                atac_raw_emb,   
                adata_adt_inputs=[adata_cite_adt, adata_asap_adt], # 
                pp_dict = ppd
        )
    
if 1:
    model.train(
        batch_size=256, training_steps=400, 
        lr=0.01, 
        log_step=50, eval_atac=False, #eval_top_k=1, eval_open=True,  
    )
else:
    # loading checkpoints
        ckpt_path = join(output_dir, 'ckpt_cite.pth')
        model.load_ckpt(ckpt_path)