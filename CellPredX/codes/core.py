import sys
import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import tables
from collections import defaultdict

import torch

import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from model import *
from dataset import ClsDataset, hvg_binz_lognorm_scale
from utils import objectview, adjust_learning_rate
from loss import L1regularization, InfoNCE,CenterLoss,CenterLoss_cos,SparseCenterLoss,DCL
from loss import CosineHead, CosineMarginHead, SoftmaxMarginHead
from sNNs import NN
from knn_classifier import kNN_approx, knn_classifier_top_k, knn_classifier_eval
from metrics import osr_evaluator

import torch.utils.data.dataloader as dataloader
#from line_profiler import LineProfiler
class CombinedModel(nn.Module):
    def __init__(self, encoder, cell):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.cell = cell
    def forward(self, x):
        embedding, attention = self.encoder(x)
        output = self.cell(embedding)
        output=nn.Softmax(dim=1)(output)
        return output
def sample_train_label(batchsizes, train_y):
    # 计算train_y中每一个样本的个数
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    class_number = len(unique_classes)
    
    # 计算每个类别应选取的样本数
    samples_per_class = batchsizes // class_number
    
    need_index = []
    
    for cls in unique_classes:
        cls_indices = np.where(train_y == cls)[0]
        if len(cls_indices) > samples_per_class:
            # 随机选择samples_per_class个样本
            selected_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            # 如果样本数少于或等于samples_per_class，选择所有样本
            selected_indices = cls_indices
        need_index.extend(selected_indices)
    
    # 如果need_index小于batchsizes，随机选择差值的样本数，但是不与前面重复
    if len(need_index) < batchsizes:
        remaining_indices = np.setdiff1d(np.arange(len(train_y)), need_index)
        additional_indices = np.random.choice(remaining_indices, batchsizes - len(need_index), replace=False)
        need_index.extend(additional_indices)
    
    return need_index



class BuildCellpredX(object):
    def __init__(self, 
                encoder_type='linear', n_latent=20, bn=False, dr=0.2, 
                l1_w=0.1, ortho_w=0.1, 
                cont_w=0.0, cont_tau=0.4, cont_cutoff=0.,
                align_w=0.0, align_p=0.8, align_cutoff=0.,mycenter_w=0.1, center_cutoff=10,
                clamp=None,
                seed=1234,n_input=2000,n_class=12,
                ):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.n_latent = n_latent
        self.encoder_type = encoder_type
        self.bn = bn
        self.dr = dr
        self.l1_w = l1_w
        self.ortho_w = ortho_w
        self.cont_w = cont_w
        self.cont_tau = cont_tau
        self.cont_cutoff = cont_cutoff
        self.align_w = align_w
        self.align_p = align_p
        self.mycenter_w=mycenter_w
        self.align_cutoff = align_cutoff
        self.center_cutoff=center_cutoff
        self.clamp = clamp
        self.n_input=n_input
        self.n_class=n_class
        if self.encoder_type == 'linear':
            self.encoder = torch.nn.DataParallel(Net_encoder(self.n_input, self.n_latent).cuda())
        else:
            self.encoder = torch.nn.DataParallel(Nonlinear_encoder(self.n_input, self.n_latent, self.bn, self.dr).cuda())

        self.head  = torch.nn.DataParallel(Net_cell(self.n_latent, self.n_class).cuda())
        self.mymodel=CombinedModel(self.encoder,self.head)
        

    def preprocess(self, 
                   adata_inputs,   # list of 'anndata' object
                   atac_raw_emb,   
                   pp_dict,
                   adata_adt_inputs=None, # list of adata_adt 
                   ):
        '''
        Performing preprocess for a pair of datasets.
        '''
        rna = adata_inputs[0].copy()
        atac = adata_inputs[1].copy()
        n_rna, n_atac = rna.shape[0], atac.shape[0]
        n_feature1, n_feature2 = rna.shape[1], atac.shape[1]
        assert n_feature1 == n_feature2, 'unmatched feature dim'
        assert (rna.var_names == atac.var_names).all(), 'unmatched feature names'

        self.batch_label = pp_dict['batch_label']
        self.type_label  = pp_dict['type_label']
        

        rna, atac, hvg_total = hvg_binz_lognorm_scale(rna, atac, pp_dict['hvg_num'], pp_dict['binz'], 
                                            pp_dict['lognorm'], pp_dict['scale_per_batch']) #获取高表达基因,并且log1p
        self.hvg_total = hvg_total

        self.data_A = sps.csr_matrix(rna.X)#参考数据据
        self.data_B = sps.csr_matrix(atac.X)#查询数据集
        self.emb_B = atac_raw_emb# 这个是原始的atac的embedding

        if adata_adt_inputs is not None:
            print('Concating adt features...')
            csr_adt_a = sps.csr_matrix(adata_adt_inputs[0].X)
            self.data_A = sps.csr_matrix(sps.hstack([self.data_A, csr_adt_a]))
            csr_adt_b = sps.csr_matrix(adata_adt_inputs[1].X)
            self.data_B = sps.csr_matrix(sps.hstack([self.data_B, csr_adt_b]))

        self.n_input = self.data_A.shape[1]
        self.n_rna, self.n_atac = n_rna, n_atac
        self.meta_A = rna.obs.copy()
        self.meta_B = atac.obs.copy() 

        y_A = self.meta_A[self.type_label].values #参考数据集的标签
        y_B = self.meta_B[self.type_label].values
        self.relabel(y_A)#变成数字标签
        # self.share_mask = np.in1d(y_B, self.class_A) 
        # self.share_class_name = np.unique(y_B[self.share_mask])
        self.shuffle_data()
 
        self.get_nns(pp_dict['knn'], pp_dict['knn_by_tissue'])
        self.y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in y_B]).astype('int32')

        return self.y_id_B,self.trainlabel2id
    

    def peak_preprocess(self, 
                   adata_inputs,   # list of 'anndata' object
                   atac_raw_emb,   
                   pp_dict,
                   adata_adt_inputs=None, # list of adata_adt 
                   ):
        '''
        Performing preprocess for a pair of datasets.
        '''
        rna = adata_inputs[0].copy()
        atac = adata_inputs[1].copy()
        n_rna, n_atac = rna.shape[0], atac.shape[0]
        n_feature1, n_feature2 = rna.shape[1], atac.shape[1]
        assert n_feature1 == n_feature2, 'unmatched feature dim'
        assert (rna.var_names == atac.var_names).all(), 'unmatched feature names'

        self.batch_label = pp_dict['batch_label']
        self.type_label  = pp_dict['type_label']
        #log1p
        rna.X = np.log1p(rna.X)
        atac.X = np.log1p(atac.X)
        self.data_A = sps.csr_matrix(rna.X)#参考数据据
        self.data_B = sps.csr_matrix(atac.X)#查询数据集
        self.emb_B = atac_raw_emb# 这个是原始的atac的embedding

        if adata_adt_inputs is not None:
            print('Concating adt features...')
            csr_adt_a = sps.csr_matrix(adata_adt_inputs[0].X)
            self.data_A = sps.csr_matrix(sps.hstack([self.data_A, csr_adt_a]))
            csr_adt_b = sps.csr_matrix(adata_adt_inputs[1].X)
            self.data_B = sps.csr_matrix(sps.hstack([self.data_B, csr_adt_b]))

        self.n_input = self.data_A.shape[1]
        self.n_rna, self.n_atac = n_rna, n_atac
        self.meta_A = rna.obs.copy()
        self.meta_B = atac.obs.copy() 

        y_A = self.meta_A[self.type_label].values #参考数据集的标签
        y_B = self.meta_B[self.type_label].values
        self.relabel(y_A)#变成数字标签
        # self.share_mask = np.in1d(y_B, self.class_A) 
        # self.share_class_name = np.unique(y_B[self.share_mask])
        self.shuffle_data()
        self.get_nns(pp_dict['knn'], pp_dict['knn_by_tissue'])
        self.y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in y_B]).astype('int32')

        return self.y_id_B,self.trainlabel2id

    def relabel(self, y_A):
        self.y_A = y_A

        self.class_A = np.unique(self.y_A)
        # self.class_B = np.unique(self.y_B)

        self.trainlabel2id = {v:i for i,v in enumerate(self.class_A)}
        self.id2trainlabel = {v:k for k,v in self.trainlabel2id.items()}

        self.y_id_A = np.array([self.trainlabel2id[_] for _ in self.y_A]).astype('int32')
        # self.y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in self.y_B]).astype('int32')
        self.n_class = len(self.class_A)

    def shuffle_data(self):
        # shuffle source domain
        rand_idx_ai = np.arange(self.n_rna)
        np.random.shuffle(rand_idx_ai)
        self.data_A_shuffle = self.data_A[rand_idx_ai]
        self.meta_A_shuffle = self.meta_A.iloc[rand_idx_ai]
        self.y_A_shuffle = self.y_A[rand_idx_ai]
        self.y_id_A_shuffle = self.y_id_A[rand_idx_ai].astype('int32')
        # shuffle target domain
        random_idx_B = np.arange(self.n_atac)
        np.random.shuffle(random_idx_B)
        self.data_B_shuffle = self.data_B[random_idx_B]
        self.emb_B_shuffle = self.emb_B[random_idx_B]
        self.meta_B_shuffle = self.meta_B.iloc[random_idx_B]
        # self.y_B_shuffle = self.y_B[random_idx_B]
        # self.y_id_B_shuffle = self.y_id_B[random_idx_B].astype('int32')

    def get_nns(self, k=10, knn_by_tissue=False):
        if not knn_by_tissue:
            knn_ind = NN(self.emb_B_shuffle, query=self.emb_B_shuffle, k=k+1, metric='manhattan', n_trees=10)[:, 1:]
        else:
            assert 'tissue' in self.meta_B_shuffle.columns, 'tissue not found in metadata'
            tissue_set = self.meta_B_shuffle['tissue'].unique()
            tissue_vec = self.meta_B_shuffle['tissue'].values
            knn_ind = np.zeros((self.n_atac, k))
            for ti in tissue_set:
                ti_idx = np.where(tissue_vec==ti)[0]
                ti_knn_ind = NN(self.emb_B_shuffle[ti_idx], self.emb_B_shuffle[ti_idx], k=k+1)[:, 1:]
                knn_ind[ti_idx, :] = ti_idx[ti_knn_ind.ravel()].reshape(ti_knn_ind.shape)

        knn_ind = knn_ind.astype('int64')

        if self.type_label in self.meta_B_shuffle.columns:
            y_ = self.meta_B_shuffle[self.type_label].to_numpy()
            y_knn = y_[knn_ind.ravel()].reshape(knn_ind.shape)
                
            ratio = (y_.reshape(-1, 1) == y_knn).mean(axis=1).mean()
            print('==========************************************================')
            print('knn correct ratio = {:.4f}'.format(ratio))
            print('===========**************************************===============')

        self.knn_ind = knn_ind

    def cor(self, m):   # covariance matrix of embedding features
        m = m.t()
        fact = 1.0 / (m.size(1) - 1)
        m = m - torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        return fact * m.matmul(mt).squeeze()

    def euclidean(self, x1, x2):
        return ((x1-x2)**2).sum().sqrt()

    def non_corr(self, x):
        l = torch.mean(torch.abs(torch.triu(self.cor(x), diagonal=1)))
        return l

    def zero_center(self, x):  # control value magnitude
        l = torch.mean(torch.abs(x))
        return l

    def max_var(self, x):
        l = max_moment1(x)
        return l

    def get_pos_ind(self, ind):
        choice_per_nn_ind = np.random.randint(low=0, high=self.knn_ind.shape[1], size=ind.shape[0])
        pos_ind = self.knn_ind[ind, choice_per_nn_ind]
        return pos_ind

    def init_train(self, opt, lr, lr2, weight_decay,stepss=100):
        # feature extractor
        if self.encoder_type == 'linear':
            self.encoder = torch.nn.DataParallel(Net_encoder(self.n_input, self.n_latent).cuda()) #多个GPU 加速训练
        else:
            self.encoder = torch.nn.DataParallel(Nonlinear_encoder(self.n_input, self.n_latent, self.bn, self.dr).cuda())
        self.head  = torch.nn.DataParallel(Net_cell(self.n_latent, self.n_class).cuda())
        if opt == 'adam':
            optimizer_G = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_C = optim.Adam(self.head.parameters(), lr=lr2 if lr2 is not None else lr, weight_decay=weight_decay)
            # 定义学习率调度器
            scheduler_G = StepLR(optimizer_G, step_size=stepss, gamma=0.1)
            # 定义学习率调度器
            scheduler_C = StepLR(optimizer_C, step_size=stepss, gamma=0.1)
        elif opt == 'sgd':
            optimizer_G = optim.SGD(self.encoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            optimizer_C = optim.SGD(self.head.parameters(), lr=lr2 if lr2 is not None else lr, momentum=0.9, weight_decay=weight_decay)
            scheduler_G = StepLR(optimizer_G, step_size=stepss, gamma=0.1)
            # 定义学习率调度器
            scheduler_C = StepLR(optimizer_C, step_size=stepss, gamma=0.1)
        return optimizer_G,scheduler_G, optimizer_C,scheduler_C


    # @profile
    def train_step(
            self, 
            step, batch_size, 
            optimizer_G, optimizer_C, scheduler_G, scheduler_C,
            cls_crit, reg_crit, reg_cont, center_loss,
            log_step=100, 
            eval_atac=False, eval_top_k=1, eval_open=False,output_dir="",best_accs=0,
        ):
        self.encoder.train()
        self.head.train()
        pr_B_top_acc = 0

        N_A = self.n_rna
        N_B = self.n_atac

        issparse = sps.issparse(self.data_B) #检查是否是稀缺矩阵
        # cls_crit_per = nn.CrossEntropyLoss(reduction='none')
        # # A input  # 随机抽取一部分数据作为训练数据
        # index_A_first = np.random.choice(np.arange(N_A), size=batch_size*4, replace=False)
        # x_A_first = torch.from_numpy(self.data_A_shuffle[index_A_first, :].A).float().cuda()
        # y_A_first = torch.from_numpy(self.y_id_A_shuffle[index_A_first]).long().cuda()
        # f_A_first = self.encoder(x_A_first)
        # if self.clamp:
        #     f_A_first = torch.clamp(f_A_first, min=-self.clamp, max=self.clamp)
        # p_A_first  = self.head(f_A_first)
        # loss_cls_first = cls_crit_per(p_A_first, y_A_first)
        # #对loss_cls_first进行排序，并且返回索引
        # _, indices = torch.sort(loss_cls_first, descending=True)
        # indices=indices.cpu().numpy()

        # #取最大的batch_size个样本
        #index_A = index_A_first[indices[:batch_size]]
        #index_A=sample_train_label(batch_size,self.y_id_A_shuffle)
        index_A = np.random.choice(np.arange(N_A), size=batch_size, replace=False)
        x_A = torch.from_numpy(self.data_A_shuffle[index_A, :].A).float().cuda()
        y_A = torch.from_numpy(self.y_id_A_shuffle[index_A]).long().cuda()
        index_B = np.random.choice(np.arange(N_B), size=batch_size,replace=False)
        x_B = torch.from_numpy(self.data_B_shuffle[index_B, :].A).float().cuda()
        # forward
        f_A,A = self.encoder(x_A)
        if self.clamp:
            f_A = torch.clamp(f_A, min=-self.clamp, max=self.clamp)
        p_A  = self.head(f_A) 
        f_B,B = self.encoder(x_B)
        if self.clamp:
            f_B = torch.clamp(f_B, min=-self.clamp, max=self.clamp)
        p_B = self.head(f_B)
        optimizer_G.zero_grad()
        optimizer_C.zero_grad()
        # Adapted NNDR loss
        A_center_loss = self.zero_center(f_A)
        A_corr_loss   = self.non_corr(f_A)
        A_var_loss    = self.max_var(f_A)
        B_center_loss = self.zero_center(f_B)
        B_corr_loss   = self.non_corr(f_B)
        B_var_loss    = self.max_var(f_B)

        adapted_NNDR_loss = A_center_loss+B_center_loss+A_corr_loss+B_corr_loss+B_var_loss
        # NCL loss
        cont_loss = 0
        if self.cont_w != 0 and (step>=self.cont_cutoff):
            B_pos_ind = self.get_pos_ind(index_B)
            x_B_pos = torch.from_numpy(self.data_B_shuffle[B_pos_ind, :].A).float().cuda()

            f_B_pos,B_pos = self.encoder(x_B_pos)
            if self.clamp:
                f_B_pos = torch.clamp(f_B_pos, min=-self.clamp, max=self.clamp)
            cont_loss = reg_cont(f_B, f_B_pos)+reg_cont(f_B_pos, f_B)

            #cont_loss=DCL_cont(f_B, f_B_pos)
        # # Alignment loss
        # align_loss = 0.    
        # if (self.align_w != 0) and (step >= self.align_cutoff):
        #     bs = f_B.size(0)
        #     # cosine similarity loss  
        #     f_A_norm = F.normalize(f_A, p=2, dim=1)
        #     f_B_norm = F.normalize(f_B, p=2, dim=1)
        #     f_A_norm_detach, f_B_norm_detach = f_A_norm.detach(), f_B_norm.detach()  
        #     cos_sim = torch.matmul(f_B_norm_detach, f_A_norm_detach.t())
        #     vals, inds = torch.max(cos_sim, dim=1)
        #     vals, top_B_inds = torch.topk(vals, int(bs * self.align_p))
        #     top_B_A_inds = inds[top_B_inds]  # corresponding A indices

        #     # maximize similarity between top_B_inds, top_B_A_inds
        #     f_B_norm_top = f_B_norm[top_B_inds]
        #     f_A_norm_top = f_A_norm[top_B_A_inds]
        #     align_loss = -torch.mean(torch.sum(f_A_norm_top * f_B_norm_top, dim=1))  # -cos_similarity
        # Alignment loss
        align_loss = 0.0    
        if (self.align_w != 0) and (step >= self.align_cutoff):
            bs = f_B.size(0)
    # cosine similarity loss  
            f_A_norm = F.normalize(f_A, p=2, dim=1)
            f_B_norm = F.normalize(f_B, p=2, dim=1)
            f_A_norm_detach, f_B_norm_detach = f_A_norm.detach(), f_B_norm.detach()  
            cos_sim = torch.matmul(f_B_norm_detach, f_A_norm_detach.t())
            vals, inds = torch.max(cos_sim, dim=1)
            vals, top_B_inds = torch.topk(vals, int(bs * self.align_p))
            top_B_A_inds = inds[top_B_inds]  # corresponding A indices

        # maximize similarity between top_B_inds, top_B_A_inds
            f_B_norm_top = f_B_norm[top_B_inds]
            f_A_norm_top = f_A_norm[top_B_A_inds]
            align_loss = -torch.mean(torch.sum(f_A_norm_top * f_B_norm_top, dim=1))  # -cos_similarity

        # Negative sample contrastive loss
            neg_inds = torch.randint(0, bs, (int(bs * self.align_p),))
            f_A_norm_neg = f_A_norm[neg_inds]
            f_B_norm_neg = f_B_norm[neg_inds]
            neg_cos_sim = torch.sum(f_A_norm_top * f_B_norm_neg, dim=1)
            neg_loss = torch.mean(F.relu(neg_cos_sim + 0.5))  # 0.2 is the margin

        # Combined loss
            align_loss = align_loss + neg_loss

        # Supervised classification loss
        loss_cls = cls_crit(p_A, y_A)
        #my center loss
        #rna的伪标签
        center_loss_value=0
        if step>self.center_cutoff:
            #concatenated_tensor = torch.cat((p_A, p_B), dim=0)
            probility_A = F.softmax(p_B, dim=1)
            max_probs, _ = torch.max(probility_A, dim=1)
            # 计算概率值的中位数
            #median_prob = max_probs.median()
            #计算75分位
            median_prob = max_probs.kthvalue(int(max_probs.size(0) * 0.75)).values
            # 选择概率大于等于中位数的样本索引
            selected_indices = max_probs >= median_prob
            # 根据索引选择per_labels中的样本
            selected_per_labels = torch.argmax(probility_A[selected_indices], dim=1)
            selected_f_B = f_B[selected_indices]
            all_labels=torch.cat((y_A,selected_per_labels),dim=0)
            concatenated_embedding=torch.cat((f_A, selected_f_B), dim=0)
            # print (concatenated_embedding)
            # print ("f_B shape",f_B.shape)
            # print ("A shape",A.shape)
            # print ("B shape",B[selected_indices])
            attention=torch.cat((A,B[selected_indices]),dim=0)
            #print ("sttention shape",attention.shape)
            center_loss_value = center_loss(concatenated_embedding,attention,all_labels)
        # Regularization loss
        l1_reg_loss = reg_crit(self.encoder) + reg_crit(self.head)
        loss = loss_cls + l1_reg_loss + self.ortho_w*adapted_NNDR_loss + self.cont_w*cont_loss + self.align_w*align_loss+self.mycenter_w*center_loss_value
        #loss = loss_cls + l1_reg_loss + self.cont_w*cont_loss + self.align_w*align_loss+self.mycenter_w*center_loss_value
        loss.backward()
        optimizer_G.step()
        optimizer_C.step()
        scheduler_C.step()
        scheduler_G.step()
        best_result = 0.89
        # logging info
        if not (step % log_step):
            print("step %d, loss_cls=%.3f, loss_l1_reg=%.3f, center=(%.3f, %.3f), corr=(%.3f, %.3f), var=(%.3f, %.3f), loss_cont=%.3f, loss_align=%.3f, loss_mycenter=%.3f" % \
                (step, loss_cls, l1_reg_loss, \
                 self.ortho_w*A_center_loss, self.ortho_w*B_center_loss, self.ortho_w*A_corr_loss, self.ortho_w*B_corr_loss, \
                 self.ortho_w*A_var_loss, self.ortho_w*B_var_loss, \
                 self.cont_w*cont_loss, 
                 self.align_w*(1+align_loss),
                 self.mycenter_w*center_loss_value
                )
            )
            feat_A, feat_B, head_A, head_B = self.eval(inplace=False)
            pr_A = np.argmax(head_A, axis=1)
            pr_B = np.argmax(head_B, axis=1)
            pr_B_top_k = np.argsort(-1 * head_B, axis=1)[:, :eval_top_k]
            # if cell type annotation of scATAC-seq data available
            # then, evaluate the performance
            
            if eval_atac and (self.type_label in self.meta_B.columns):
                y_B = self.meta_B[self.type_label].to_numpy()
                y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in y_B])
                share_mask = np.in1d(y_B, self.class_A)
                pr_B_top_acc = knn_classifier_eval(pr_B_top_k, y_id_B, True, share_mask)
                from sklearn.metrics import f1_score
                f1 = f1_score(y_id_B[share_mask], pr_B[share_mask], average='macro')
                if not eval_open:  # close-set eval
                    print("Overall acc={:.5f}".format(pr_B_top_acc))
                    print ("f1 score=",f1)
                    if pr_B_top_acc>best_accs:
                    #保存参数
                        state = {'encoder': self.encoder.state_dict(), 
                                'head': self.head.state_dict(),}
                        torch.save(state, os.path.join(output_dir, str(pr_B_top_acc)+".pth"))
    
                else:              # open-set eval
                    closed_score = np.max(head_B, axis=1)
                    open_score   = 1 - closed_score
                    kn_data_pr = pr_B[share_mask]
                    kn_data_gt = y_id_B[share_mask]
                    kn_data_open_score = open_score[share_mask]
                    unk_data_open_score = open_score[np.logical_not(share_mask)]
                    closed_acc, os_auroc, os_aupr, oscr = osr_evaluator(kn_data_pr, kn_data_gt, kn_data_open_score, unk_data_open_score)
        return loss_cls.item(), pr_B_top_acc

    def train(self, 
            opt='sgd', 
            batch_size=500, training_steps=2000, 
            lr=0.001, lr2=None, weight_decay=5e-4,
            log_step=100, eval_atac=False, eval_top_k=1, eval_open=False,output_dir='/data/yan_code/scDWL/outputs/',
            ):
        # torch.manual_seed(1)
        begin_time = time.time()    

        # init model
        optimizer_G,scheduler_G, optimizer_C,scheduler_C = self.init_train(opt, lr, lr2, weight_decay,stepss=training_steps/2)

        reg_crit = L1regularization(self.l1_w).cuda()
        reg_cont = InfoNCE(batch_size, self.cont_tau).cuda()
        #reg_cont=DCL(self.cont_tau).cuda()
        #DCL_cont=DCL(self.cont_tau).cuda()
        # cls_crit = nn.CrossEntropyLoss(reduction='none').cuda()
        cls_crit = nn.CrossEntropyLoss().cuda()
        #mycenter_loss=CenterLoss(num_classes=self.n_class, feat_dim=self.n_latent,use_gpu=True)
        #mycenter_loss=CenterLoss_cos(num_classes=self.n_class, feat_dim=self.n_latent,use_gpu=True)
        mycenter_loss=SparseCenterLoss(num_classes=self.n_class, feat_dim=self.n_latent).cuda()

        self.loss_cls_history = []
        best_acc=0
        for step in range(training_steps):
            loss_cls,best_results = self.train_step( 
                step, batch_size,
                optimizer_G=optimizer_G, optimizer_C=optimizer_C, scheduler_G=scheduler_G, scheduler_C=scheduler_C,
                cls_crit=cls_crit, reg_crit=reg_crit, reg_cont=reg_cont, center_loss=mycenter_loss,
                log_step=log_step, 
                eval_atac=eval_atac, eval_top_k=eval_top_k, eval_open=eval_open,output_dir=output_dir,best_accs=best_acc,
            )
            best_acc=best_results

            self.loss_cls_history.append(loss_cls)

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

    def analyze_feature_importances_for_all_cell_types(self,model, data_loader):
        CombinedModelInstance = model
        CombinedModelInstance.eval()
        CombinedModelInstance.cuda()
        ig = IntegratedGradients(CombinedModelInstance)
    # 假设你知道类别的总数，或者你可以从数据中获取
        num_classes = self.n_class  # 确保这是正确的类别数
    # 存储每个类别的特征重要性统计
        all_features = {}
        for cell_type in range(num_classes):
            feature_counts = defaultdict(int)
        # 遍历数据加载器中的所有数据
            for inputs, labels in data_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = CombinedModelInstance(inputs)
                preds = outputs.argmax(dim=1)
            #     print (preds)
            #     print (labels)
            # #计算preds和labels的准确率
            #     accuracy = (preds == labels).sum().item() / len(labels)
            #     print (accuracy)
            # 筛选出特定细胞类型的正确预测
                correct = (preds == labels) & (labels == cell_type)
                correct_inputs = inputs[correct]
                correct_labels = labels[correct]
            # 对每个正确的样本计算特征贡献
                for i in range(correct_inputs.size(0)):
                    attributions = ig.attribute(correct_inputs[i:i+1], target=correct_labels[i])
                # 获取贡献度最高的100个特征
                    top_features = attributions.abs().flatten().topk(100).indices
                    for feature in top_features:
                        feature_counts[feature.item()] += 1
        # 获取出现频率最高的50个特征
            top_50_features = Counter(feature_counts).most_common(50)
            cell_type_str=self.id2trainlabel[cell_type]
            print (self.id2trainlabel[cell_type])

            all_features[cell_type] = [(self.hvg_total[feature], count) for feature, count in top_50_features]
            print ("Top 50 Features for Cell Type"+ cell_type_str+":",all_features[cell_type])
            # for gene_name, count in all_features[cell_type]:
            #     print(f"Feature: {gene_name}, Count: {count}")
        return all_features

    def eval(self, batch_size=500, inplace=False):
        # test loader
        src_ds = ClsDataset(self.data_A, self.y_id_A, binz=False, train=False)   # for evaluation
        tgt_ds = ClsDataset(self.data_B, np.ones(self.n_atac, dtype='int32'), binz=False, train=False)
        self.src_dl = dataloader.DataLoader(src_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False) 
        self.tgt_dl = dataloader.DataLoader(tgt_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

        self.encoder.eval()
        self.head.eval()
        self.mymodel.eval()
        feat_A, head_A = [], []
        for x, y in self.src_dl:
            x = x.cuda()
            z_A,sss= self.encoder(x)
            if self.clamp:
                z_A = torch.clamp(z_A, min=-self.clamp, max=self.clamp)
            h_A = nn.Softmax(dim=1)(self.head(z_A))
            feat_A.append(z_A.detach().cpu().numpy())
            head_A.append(h_A.detach().cpu().numpy())
        feat_B, head_B = [], []
        for x, y in self.tgt_dl:
            x = x.cuda()
            z_B,ss = self.encoder(x)
            if self.clamp:
                z_B = torch.clamp(z_B, min=-self.clamp, max=self.clamp)
            h_B = nn.Softmax(dim=1)(self.head(z_B))
            feat_B.append(z_B.detach().cpu().numpy())
            head_B.append(h_B.detach().cpu().numpy())

        feat_A, feat_B = np.vstack(feat_A), np.vstack(feat_B)
        head_A, head_B = np.vstack(head_A), np.vstack(head_B)
        #all_features = self.analyze_feature_importances_for_all_cell_types(self.mymodel,self.tgt_dl)
        ###############分析特征############
        # print("Top 50 most important features across correctly predicted samples:")
        # print(top_features)
        if inplace:
            self.feat_A = feat_A
            self.feat_B = feat_B
            self.head_A = head_A
            self.head_B = head_B
            self.feat_AB = np.vstack([feat_A, feat_B])
            self.head_AB = np.vstack([head_A, head_B])
        else:
            return feat_A, feat_B, head_A, head_B
        
    def load_ckpt(self, path):
        
        self.encoder.load_state_dict(torch.load(path)['encoder'])
        self.head.load_state_dict(torch.load(path)['head'])
        print(f'loaded checkpoints from {path}')

    def annotate(self, label_prop=False, prop_knn=10):
        try:
            self.head_B
        except:
            self.eval(inplace=True)

        atac_pr = np.argmax(self.head_B, axis=1)
        if label_prop:
            atac_pr = kNN_approx(self.feat_B, self.feat_B, atac_pr, n_sample=None, knn=prop_knn)

        atac_pr = np.array([self.id2trainlabel[_] for _ in atac_pr])
        return atac_pr
    def feature_importance(self):
        try:
            self.head_B
        except:
            self.eval(inplace=True)
        data_loader = self.tgt_dl
        all_features = self.analyze_feature_importances_for_all_cell_types(data_loader)
        return all_features


def max_moment0(feats):
    loss = 1 / torch.mean(torch.abs(feats - torch.mean(feats, dim=0)))
    return loss

def max_moment1(feats):
    loss = 1 / torch.mean(   
            torch.abs(feats - torch.mean(feats, dim=0)))
    return loss

def inter_class_dist(v_cls, feats):
    cls_set = np.unique(v_cls)
    cls_centers = []
    for i, ci in enumerate(cls_set):
        ci_mask = v_cls == ci
        cls_centers.append(feats[ci_mask].mean(axis=0))
    cls_centers = np.vstack(cls_centers)

    inter_dist = pairwise_distances(cls_centers)
    ds = np.tril_indices(inter_dist.shape[0], k=-1)  # below the diagonal
    v = inter_dist[ds].mean()
    return v

def intra_class_dist(v_cls, feats):
    cls_set = np.unique(v_cls)
    cls_vars = []
    for i, ci in enumerate(cls_set):
        ci_mask = v_cls == ci
        cent = feats[ci_mask].mean(axis=0, keepdims=True)
        ci_var = pairwise_distances(cent, feats[ci_mask]).mean()
        cls_vars.append(ci_var)
    intra_var = np.mean(cls_vars)
    return intra_var

def measure_var(v_cls, feats, probs, l2norm=False):
    if l2norm:
        feats = normalize(feats, axis=1)  

    inter_var = inter_class_dist(v_cls, feats)
    intra_var = intra_class_dist(v_cls, feats)
    total_var = inter_var / intra_var
    return inter_var, intra_var, total_var

def save_ckpts(output_dir, model, step):
    state = {
            'encoder': model.encoder.state_dict(), 
            'head': model.head.state_dict(), 
            }
    torch.save(state, os.path.join(output_dir, f"ckpt_{step}.pth"))


