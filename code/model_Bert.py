#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
from random import sample
import torch
import torch.nn as nn
from torch import optim
import math
import numpy as np
from transformers import BertTokenizer, BertModel, DebertaTokenizer, BertConfig, DebertaModel, AutoTokenizer, AutoModel, \
    AutoModelForSequenceClassification

from test_performance import cluster_test, HAC_getClusters, trans_embed, cluster_result, seed_accuracy, score_result
from helper import checkFile
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def reg_loss_func( current_vec, prev_vec, lambda_reg=0.1):

    #  L2距离平方
    time_loss = torch.sum((current_vec - prev_vec) ** 2, dim=-1)

    # 当前向量L2正则化
    reg_term = torch.sum(current_vec ** 2, dim=-1)

    # 组合损失
    total_loss = time_loss + lambda_reg * reg_term
    return total_loss.mean()  # 返回批次平均损失

class ContrastiveLossWithHardNegatives(nn.Module):
    def __init__(self, temperature=0.1, hard_negative_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, anchor_embeddings, positive_embeddings, hard_negative_embeddings=None):
        """
        anchor_embeddings: embeddings of anchor samples [batch_size, embedding_dim]
        positive_embeddings: embeddings of positive samples [batch_size, embedding_dim]
        hard_negative_embeddings: optional, embeddings of hard negative samples [batch_size, num_hard_negatives, embedding_dim]
        """
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute similarity with positives
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1) / self.temperature
        pos_sim = pos_sim.unsqueeze(-1)  # [batch_size, 1]

        # Compute similarity with in-batch negatives
        in_batch_neg_sim = torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature

        # Initialize negative similarities with in-batch negatives
        neg_sims = in_batch_neg_sim

        # Add hard negatives if provided
        if hard_negative_embeddings is not None:
            hard_negative_embeddings = F.normalize(hard_negative_embeddings, p=2, dim=-1)

            # Compute similarity with hard negatives
            hard_neg_sim = torch.matmul(
                anchor_embeddings.unsqueeze(1),  # [batch_size, 1, embedding_dim]
                hard_negative_embeddings.transpose(-1, -2)  # [batch_size, embedding_dim, num_hard_negatives]
            ).squeeze(1) / self.temperature  # [batch_size, num_hard_negatives]

            # Weight the hard negatives
            hard_neg_sim = hard_neg_sim * self.hard_negative_weight

            # Combine in-batch negatives and hard negatives
            neg_sims = torch.cat([neg_sims, hard_neg_sim], dim=1)

        # Construct logits: [pos_sim, neg_sim1, neg_sim2, ...]
        logits = torch.cat([pos_sim, neg_sims], dim=1)

        # Labels are 0 (the positive is always at position 0)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor_embeddings.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

class ContrastiveLossWithInBatchNegatives(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embeddings, positive_embeddings):
        """
        anchor_embeddings: embeddings of anchor samples [batch_size, embedding_dim]
        positive_embeddings: embeddings of positive samples [batch_size, embedding_dim]
        """
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature

        # Labels are the diagonal elements (each anchor's positive is on the diagonal)
        labels = torch.arange(logits.size(0), device=anchor_embeddings.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


def hinge_loss(sub_emb, rel_emb, obj_emb, neg_embs, gamma):
    # 相似度计算，'tail-batch',获取 stru_loss
    pos_score = (sub_emb + rel_emb) - obj_emb
    pos_score = torch.norm(pos_score, p=1, dim=2)
    # pos_score = pos_score.repeat(1, neg_embs.size(1))

    neg_score = (sub_emb + rel_emb) - neg_embs
    neg_score = torch.norm(neg_score, p=1, dim=2)

    max_values, max_indices = torch.max(neg_score, dim=1, keepdim=True)
    # gamma = torch.full((1, neg_embs.size(1)), float(gamma)).cuda()

    err = pos_score - max_values + gamma
    max_err = err.clamp(0)
    return max_err




class BertEmbeddingModel(nn.Module):
    def __init__(self, max_length, target_dim):
        super(BertEmbeddingModel, self).__init__()
        self.max_length = max_length
        self.target_dim = target_dim
        self.bert_tri = BertModel.from_pretrained('../plm')
        self.tokenizer_tri = BertTokenizer.from_pretrained('../plm')
        self.tokenizer_tri.add_tokens(['[S]', '[R]', '[choice]'])
        self.bert_tri.resize_token_embeddings(len(self.tokenizer_tri))

        self.bert_sub = BertModel.from_pretrained('../plm')
        self.tokenizer_sub = BertTokenizer.from_pretrained('../plm')
        self.tokenizer_sub.add_tokens(['[context]'])
        self.bert_sub.resize_token_embeddings(len(self.tokenizer_sub))

        self.dense = nn.Linear(768, target_dim)
        # self.dense_sub = nn.Linear(768, target_dim)




    def forward(self, tri_sequences,sub_sequences,masks):

        masks = torch.tensor(masks).unsqueeze(-1).cuda()

        tri_batch_tokenized = self.tokenizer_tri.batch_encode_plus(tri_sequences, add_special_tokens=False,
                                                           max_length=self.max_length,
                                                           padding="max_length",
                                                           truncation=True)
        tri_input_ids = torch.tensor(tri_batch_tokenized['input_ids']).cuda()
        tri_attention_mask = torch.tensor(tri_batch_tokenized['attention_mask']).cuda()

        tri_bert_output = self.bert_tri(tri_input_ids, attention_mask=tri_attention_mask)['last_hidden_state']

        special_emb = masks.float() * tri_bert_output
        non_zero_rows = special_emb.sum(dim=-1) != 0
        special_embs = special_emb[non_zero_rows].reshape(tri_bert_output.size(0), -1, tri_bert_output.size(2))

        special_embs = self.dense(special_embs)
        sub_emb = special_embs[:, :1, :]
        rel_emb = special_embs[:, 1:2, :]
        obj_emb = special_embs[:, 2:3, :]
        neg_embs = special_embs[:, 3:, :]

        # sub_emb = self.dense(special_embs[:, :1, :])
        # rel_emb = self.dense(special_embs[:, 1:2, :])
        # obj_emb = self.dense(special_embs[:, 2:3, :])
        # neg_embs = self.dense(special_embs[:, 3:, :])


        if sub_sequences is None:
            sub_emb_context = None
        else:
            sub_batch_tokenized = self.tokenizer_sub.batch_encode_plus(sub_sequences, add_special_tokens=False,
                                                               max_length=self.max_length,
                                                               padding="max_length",
                                                               truncation=True)
            sub_input_ids = torch.tensor(sub_batch_tokenized['input_ids']).cuda()
            sub_attention_mask = torch.tensor(sub_batch_tokenized['attention_mask']).cuda()
            sub_bert_output = self.bert_sub(sub_input_ids, attention_mask=sub_attention_mask)
            sub_emb_context = self.dense(sub_bert_output[0][:, 0, :])
            # sub_emb_context = sub_bert_output[0][:, 0, :]

        return sub_emb, rel_emb, obj_emb, neg_embs, sub_emb_context


    def forward_seed(self, input1, input2, hard_negs):
        head_batch_tokenized = self.tokenizer_sub.batch_encode_plus(input1, add_special_tokens=False,
                                                                   max_length=self.max_length,
                                                                   padding="max_length",
                                                                   truncation=True)
        head_input_ids = torch.tensor(head_batch_tokenized['input_ids']).cuda()
        head_attention_mask = torch.tensor(head_batch_tokenized['attention_mask']).cuda()

        head_bert_output = self.bert_sub(head_input_ids, attention_mask=head_attention_mask)
        head_bert_output = self.dense(head_bert_output[0][:, 0, :])
        # head_bert_output = head_bert_output[0][:, 0, :]


        tail_batch_tokenized = self.tokenizer_sub.batch_encode_plus(input2, add_special_tokens=False,
                                                                   max_length=self.max_length,
                                                                   padding="max_length",
                                                                   truncation=True)
        tail_input_ids = torch.tensor(tail_batch_tokenized['input_ids']).cuda()
        tail_attention_mask = torch.tensor(tail_batch_tokenized['attention_mask']).cuda()

        tail_bert_output = self.bert_sub(tail_input_ids, attention_mask=tail_attention_mask)
        tail_bert_output = self.dense(tail_bert_output[0][:, 0, :])
        # tail_bert_output = tail_bert_output[0][:, 0, :]


        hard_neg_embeds = []
        for idx, data in enumerate(hard_negs):
            negs_batch_tokenized = self.tokenizer_sub.batch_encode_plus(data, add_special_tokens=False,
                                                                    max_length=self.max_length,
                                                                    padding="max_length",
                                                                    truncation=True)

            negs_input_ids = torch.tensor(negs_batch_tokenized['input_ids']).cuda()
            negs_attention_mask = torch.tensor(negs_batch_tokenized['attention_mask']).cuda()

            negs_bert_output = self.bert_sub(negs_input_ids, attention_mask=negs_attention_mask)
            negs_bert_output = self.dense(negs_bert_output[0][:, 0, :])
            # negs_bert_output = negs_bert_output[0][:, 0, :]


            hard_neg_embeds.append(negs_bert_output)

        hard_neg_embeds = torch.stack(hard_neg_embeds, dim=0)


        return head_bert_output, tail_bert_output,  hard_neg_embeds



    def get_embed(self,input):
        sub_batch_tokenized = self.tokenizer_sub.batch_encode_plus(input,
                                                               max_length=self.max_length,
                                                               padding="max_length",
                                                               truncation=True)
        sub_input_ids = torch.tensor(sub_batch_tokenized['input_ids']).cuda()
        sub_attention_mask = torch.tensor(sub_batch_tokenized['attention_mask']).cuda()

        sub_bert_output = self.bert_sub(sub_input_ids, attention_mask=sub_attention_mask)
        sub_bert_output = self.dense(sub_bert_output[0][:, 0, :])
        # sub_bert_output = sub_bert_output[0][:, 0, :]

        return sub_bert_output


class DynamicAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()  # 存储动态添加的层
        self.layer_order = []  # 记录层顺序

    def add_layer(self, name, layer):
        """添加可训练层"""
        self.layers[name] = layer
        self.layer_order.append(name)

    def forward(self, *embs, name, task_num):
        if task_num>0:
            embs_mlp = [self.layers[name](e) for e in embs]
            return embs_mlp
        else:
            return embs

    def get_embed(self, x, name):
        x = self.layers[name](x)
        return x


class Bert_Model(object):
    def __init__(self, params, snapshot_side_infos, true_ent2clust, true_clust2ent):
        self.p = params
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.snapshot_side_infos = snapshot_side_infos
        self.batch_size = 16
        self.seed_batch_size = 6
        self.epochs = 5
        self.lr = 1e-5
        self.seed_epochs = 1
        self.stru_epochs = 1
        self.reg_epochs = 1
        self.dka_epochs = 1
        self.mlp_lr = 1e-4
        self.max_length = params.max_length
        self.num_tasks = 5
        self.gamma = params.gamma

    def encode(self):
        global_sub_embeddings = dict()
        Adapter_cross = DynamicAdapter().cuda()
        Adapter_np = DynamicAdapter().cuda()
        bert_embedding_model = BertEmbeddingModel(self.max_length, target_dim=768).cuda()
        cluster_results = []
        pre_np_contexts = dict()

        for task_num in range( self.num_tasks):
            if task_num > 0:
                pre_np_contexts.update(self.snapshot_side_infos[task_num - 1].sub2context)
                crr_np_contexts = self.snapshot_side_infos[task_num].sub2context
                for key in pre_np_contexts:
                    if key in crr_np_contexts:

                        parts = crr_np_contexts[key].split("[context] ", 1)
                        crr_np_contexts[key] = pre_np_contexts[key] + parts[1]

                self.snapshot_side_infos[task_num].sub2context = crr_np_contexts




            np_seeds = pickle.load(open('../'+ self.p.dataset + f'/seeds_{task_num}', 'rb'))
            sub_cansseed_pair_list = filter_tuples(np_seeds, self.snapshot_side_infos[task_num].sub_list)

            all_sub_seeds = swap_and_deduplicate(sub_cansseed_pair_list)

            cansseed_cluster = cluster_tuples(sub_cansseed_pair_list)

            negatives_dict = sample_hard_negatives(cansseed_cluster,self.p.hard_negs)


            train_head_seeds_context, train_tail_seeds_context = [], []
            hard_negs_context = []
            for t in all_sub_seeds:
                tmp_list = []
                head = self.snapshot_side_infos[task_num].sub2context[t[0]]
                tail = self.snapshot_side_infos[task_num].sub2context[t[1]]
                for i in range(len(negatives_dict[t[0]])):
                    tmp_list.append(self.snapshot_side_infos[task_num].sub2context[negatives_dict[t[0]][i]])
                hard_negs_context.append(tmp_list)

                train_head_seeds_context.append(head)
                train_tail_seeds_context.append(tail)

            seed_batch_count2 = math.ceil(len(train_head_seeds_context) / self.seed_batch_size)
            print('seed_batch_count:', seed_batch_count2)

            batch_head_seeds_context, batch_tail_seeds_context = [], []
            batch_hard_negs_context = []
            for i in range(seed_batch_count2):
                batch_head_seeds_context.append(train_head_seeds_context[i * self.seed_batch_size:(i + 1) * self.seed_batch_size])
                batch_tail_seeds_context.append(train_tail_seeds_context[i * self.seed_batch_size:(i + 1) * self.seed_batch_size])
                batch_hard_negs_context.append(hard_negs_context[i * self.seed_batch_size:(i + 1) * self.seed_batch_size])


    ##############################################################
            clean_sub_sequences = []
            clean_subs = []
            pre_np_embs = []
            crr_nps = []
            for sub in self.snapshot_side_infos[task_num].sub_list:
                clean_sub_sequence = self.snapshot_side_infos[task_num].sub2context[sub]
                clean_sub_sequences.append(clean_sub_sequence)
                clean_subs.append(sub)
                if sub in global_sub_embeddings.keys():
                    crr_nps.append(self.snapshot_side_infos[task_num].sub2context[sub])
                    pre_np_embs.append(global_sub_embeddings[sub])

            clean_batch_count = math.ceil(len(clean_sub_sequences) / self.batch_size)
            batch_reg = math.ceil(len(crr_nps) / self.batch_size)
            print('clean_batch_count:', clean_batch_count)

            batch_clean_sub_sequence = []
            batch_clean_sub = []
            batch_crr_nps = []
            batch_pre_np_embs = []
            # batch_sub = []
            for i in range(clean_batch_count):
                batch_clean_sub_sequence.append(clean_sub_sequences[i * self.batch_size:(i + 1) * self.batch_size])
                batch_clean_sub.append(clean_subs[i * self.batch_size:(i + 1) * self.batch_size])

            for i in range(batch_reg):
                batch_crr_nps.append(crr_nps[i * self.batch_size:(i + 1) * self.batch_size])
                batch_pre_np_embs.append(pre_np_embs[i * self.batch_size:(i + 1) * self.batch_size])

      ###########################################################################################      # #
            train_triple_sequences = []
            train_masks = []
            train_contexts = []
            for tri in set(self.snapshot_side_infos[task_num].trpIds):
                triple_sequence = self.snapshot_side_infos[task_num].triple2sequence[tri]
                train_triple_sequences.append(triple_sequence)

                mask = self.snapshot_side_infos[task_num].triple2sequence_mask[tri]
                train_masks.append(mask+[0]*(self.max_length-len(mask)))


                context = self.snapshot_side_infos[task_num].sub2context[self.snapshot_side_infos[task_num].id2sub[tri[0]]]
                train_contexts.append(context)


            batch_count = math.ceil(len(train_contexts) / self.batch_size)
            print('batch_count:', batch_count)

            batch_triple_sequence = []
            batch_train_masks = []
            batch_train_contexts = []
            for i in range(batch_count):
                batch_triple_sequence.append(train_triple_sequences[i * self.batch_size:(i + 1) * self.batch_size])
                batch_train_masks.append(train_masks[i * self.batch_size:(i + 1) * self.batch_size])
                batch_train_contexts.append(train_contexts[i * self.batch_size:(i + 1) * self.batch_size])

            if task_num == 0:
                optimizer = torch.optim.AdamW(bert_embedding_model.parameters(), lr=self.lr)
                bert_embedding_model.train()
            else:
                P_adapter_cross = list(filter(lambda p: p.requires_grad, Adapter_cross.parameters()))
                P_adapter_np = list(filter(lambda p: p.requires_grad, Adapter_np.parameters()))
                optimizer = torch.optim.AdamW(P_adapter_cross+ P_adapter_np, lr=self.mlp_lr)
                Adapter_cross.train()
                Adapter_np.train()


            print(f"task_num_{task_num}:  learning start.")
            start_time_train = time.perf_counter()



            cl_loss_func_inbatch = ContrastiveLossWithInBatchNegatives(temperature=self.p.temperature)
            cl_loss_func_hard = ContrastiveLossWithHardNegatives(temperature=self.p.temperature)

            for epoch in range(self.epochs):

                for stru_epoch in range(self.stru_epochs):
                    avg_stru_loss = 0
                    for i in range(batch_count):
                        sub_emb, rel_emb, obj_emb, neg_embs, _ = bert_embedding_model(batch_triple_sequence[i], None,
                                                                                      batch_train_masks[i])
                        embs_mlp = Adapter_cross(sub_emb, rel_emb, obj_emb, neg_embs, name=f'adapter_cross_{task_num}',
                                                 task_num=task_num)

                        stru_loss = hinge_loss(embs_mlp[0], embs_mlp[1], embs_mlp[2], embs_mlp[3], self.gamma).sum()

                        print("batch_num: %d, stru_loss: %.4f" % (i, stru_loss))
                        optimizer.zero_grad()
                        stru_loss.backward()
                        # nn.utils.clip_grad_norm_(bert_embedding_model.parameters(), 1.0)
                        optimizer.step()
                        avg_stru_loss += stru_loss.item()
                        if i == (batch_count - 1):
                            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                            print(real_time, "Epoch: %d, stru_loss: %.4f" % (stru_epoch, avg_stru_loss))
                            print(real_time, "Epoch: %d" % (epoch))

                for seed_epoch in range(self.seed_epochs):
                    avg_epoch_seed_loss = 0
                    for i in range(seed_batch_count2):
                        head_emb, tail_emb, hard_neg_embs = bert_embedding_model.forward_seed(batch_head_seeds_context[i], batch_tail_seeds_context[i], batch_hard_negs_context[i])
                        embs_mlp = Adapter_np(head_emb, tail_emb, hard_neg_embs,name = f'adapter_np_{task_num}', task_num = task_num)
                        seed_loss = cl_loss_func_hard(embs_mlp[0], embs_mlp[1], embs_mlp[2])
                        print("seed_batch_num: %d, seed_Loss: %.4f" % (i, seed_loss))
                        optimizer.zero_grad()
                        seed_loss.backward()
                        # nn.utils.clip_grad_norm_(bert_embedding_model.parameters(), 1.0)
                        optimizer.step()
                        avg_epoch_seed_loss += seed_loss.item()
                        if i == (seed_batch_count2 - 1):
                            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                            print(real_time, "seed_Epoch: %d, Seed_Loss: %.4f" % (seed_epoch, avg_epoch_seed_loss))
                            print(real_time, "Epoch: %d" % (epoch))



                for dka_epoch in range(self.dka_epochs):
                    avg_epoch_dka_loss = 0
                    for i in range(batch_count):

                        sub_emb, rel_emb, obj_emb, neg_embs, sub_emb_context = bert_embedding_model(batch_triple_sequence[i],batch_train_contexts[i],batch_train_masks[i])
                        embs_mlp = Adapter_cross(sub_emb, rel_emb, obj_emb, neg_embs,name = f'adapter_cross_{task_num}', task_num = task_num)

                        sub_context_mlp = Adapter_np(sub_emb_context, name=f'adapter_np_{task_num}', task_num=task_num)


                        dka_loss = cl_loss_func_inbatch(sub_context_mlp[0], embs_mlp[0].squeeze(dim=1))



                        print( "batch_num: %d, dka_loss: %.4f" % (i, dka_loss))
                        optimizer.zero_grad()
                        dka_loss.backward()
                        # nn.utils.clip_grad_norm_(bert_embedding_model.parameters(), 1.0)
                        optimizer.step()
                        avg_epoch_dka_loss += dka_loss.item()
                        if i == (batch_count - 1):
                            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                            print(real_time, "dka_epoch: %d, dka_loss: %.4f" % (dka_epoch, avg_epoch_dka_loss))
                            print(real_time, "Epoch: %d" % (epoch))

                if task_num > 0:
                    for reg_epoch in range(self.reg_epochs):
                        avg_epoch_reg_loss = 0
                        for i in range(batch_reg):

                            sub_emb = bert_embedding_model.get_embed(batch_crr_nps[i])
                            sub_emb = Adapter_np.get_embed(sub_emb, f'adapter_np_{task_num}')
                            reg_loss = reg_loss_func(sub_emb,torch.tensor(batch_pre_np_embs[i]).cuda(), lambda_reg = self.p.lambda_reg)

                            print("reg_batch_num: %d, Loss: %.4f" % (i, reg_loss))
                            optimizer.zero_grad()
                            reg_loss.backward()
                            # nn.utils.clip_grad_norm_(bert_embedding_model.parameters(), 1.0)
                            optimizer.step()
                            avg_epoch_reg_loss += reg_loss.item()
                            if i == (batch_reg - 1):
                                real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                                print(real_time, "reg_Epoch: %d, reg_Loss: %.4f" % (reg_epoch, avg_epoch_reg_loss))
                                print(real_time, "Epoch: %d" % (epoch))




            end_time_train = time.perf_counter()
            elapsed = end_time_train - start_time_train

            print("\n" + "=" * 50)
            print(f"✅ **任务-{task_num}**训练完成！")
            print(f"⏱️  精确耗时: {elapsed:.6f} 秒")
            print(f"🕒 格式化耗时: {format_time(elapsed)}")
            print("=" * 50)


            task_sub_embeddings = {}
            if task_num ==0:
                bert_embedding_model.eval()
                with torch.no_grad():
                    for i in range(clean_batch_count):
                        sub_embeds = bert_embedding_model.get_embed(batch_clean_sub_sequence[i])
                        sub_embeds = sub_embeds.detach().cpu().numpy()
                        for j,sub in enumerate(batch_clean_sub[i]):
                            task_sub_embeddings[sub] = sub_embeds[j]
            else:
                bert_embedding_model.eval()
                Adapter_np.eval()
                with torch.no_grad():
                    for i in range(clean_batch_count):
                        sub_embeds = bert_embedding_model.get_embed(batch_clean_sub_sequence[i])
                        sub_embeds = Adapter_np.get_embed(sub_embeds,f'adapter_np_{task_num}')
                        sub_embeds = sub_embeds.detach().cpu().numpy()
                        for j, sub in enumerate(batch_clean_sub[i]):
                            task_sub_embeddings[sub] = sub_embeds[j]

            global_sub_embeddings.update(task_sub_embeddings)

            cluster_embeddings = []
            for _, sub in enumerate(self.snapshot_side_infos[task_num].sub_list_test):
                cluster_embeddings.append(global_sub_embeddings[sub])


            start_time_cluster = time.perf_counter()
            avg_f1= cluster_result(self.p, self.snapshot_side_infos[task_num], cluster_embeddings, self.true_ent2clust[task_num],
                               self.true_clust2ent[task_num], mode='semantic_embeddings')

            end_time_cluster = time.perf_counter()
            elapsed = end_time_cluster - start_time_cluster

            print("\n" + "=" * 50)
            print(f"✅ **任务{task_num}**聚类完成！")
            print(f"⏱️  精确耗时: {elapsed:.6f} 秒")
            print(f"🕒 格式化耗时: {format_time(elapsed)}")
            print("=" * 50)

            cluster_results.append(avg_f1)


            print("*******************************************")
            print(f"task_num_0-{task_num}:  learning finished.")
            print("avg_f1: ", avg_f1)
            # print("max_epoch: ",max_epoch)
            print("*******************************************")

            # snapshot_0训练完成，保存BERT参数。
            if task_num == 0:
                os.makedirs('../saved_models', exist_ok=True)
                torch.save(bert_embedding_model.state_dict(), '../saved_models/bert_freezed_weights.pth')

                for param in bert_embedding_model.parameters():  # 冻结bert层
                    param.requires_grad = False

                mlp = nn.Sequential(
                    # nn.LayerNorm(768),
                    nn.Linear(768, 384),
                    # nn.ReLU(inplace=True),
                    nn.Linear(384, 768)

                ).cuda()
                Adapter_cross.add_layer(f'adapter_cross_{task_num+1}', mlp)
                Adapter_np.add_layer(f'adapter_np_{task_num + 1}', mlp)


            else:
                torch.save(Adapter_cross.state_dict(), '../saved_models/' + f'adapter_cross_{task_num}_weights.pth')
                torch.save(Adapter_np.state_dict(), '../saved_models/' + f'adapter_np_{task_num}_weights.pth')


                if task_num < self.num_tasks-1:
                    for param in Adapter_cross.parameters():
                        param.requires_grad = False

                    for param in Adapter_np.parameters():
                        param.requires_grad = False


                    # new adapter for incremental
                    mlp = nn.Sequential(
                        # nn.LayerNorm(768),
                        nn.Linear(768, 384),
                        # nn.ReLU(inplace=True),
                        nn.Linear(384, 768)

                    ).cuda()
                    Adapter_cross.add_layer(f'adapter_cross_{task_num+1}', mlp)
                    Adapter_np.add_layer(f'adapter_np_{task_num+1}', mlp)
                    # 按层分类显示冻结/未冻结状态
                    trainable_params = []
                    frozen_params = []

                    for name, param in Adapter_np.named_parameters():
                        if param.requires_grad:
                            trainable_params.append(name)
                        else:
                            frozen_params.append(name)

                    print("------ 可训练层 ------")
                    for name in trainable_params:
                        print(name)

                    print("\n------ 冻结层 ------")
                    for name in frozen_params:
                        print(name)

                    print(f"\n总结: 可训练层 {len(trainable_params)} 个, 冻结层 {len(frozen_params)} 个")


        for i in range(len(cluster_results)):
            # print("*******************************************")
            print(f"task_num_0-{i} avg_f1: {cluster_results[i]}")
            # print("*******************************************")

        return global_sub_embeddings

