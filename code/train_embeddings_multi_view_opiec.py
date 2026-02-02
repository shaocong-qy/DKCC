import  itertools, pickle, time
from helper import *
from model_Bert import Bert_Model



def get_seed_pair_list(embedding, side_info, k):
    # 2138*5
    sub_len = len(side_info.sub_list)
    topks, topk_cosines, cosines = getTopk(embedding, k)
    seed_pair_cos = dict()
    seed_pair_list_uni = set()
    seed_pair_list = set()
    for i in range(sub_len):
        for j in range(k):
            pair = (i, topks[i][j])
            if i < topks[i][j]:
                pair_uni = (i, topks[i][j])
            else:
                pair_uni = (topks[i][j], i)
            seed_pair_list.add(pair)
            seed_pair_list_uni.add(pair_uni)

        for sub in range(i + 1, sub_len):
            seed_pair = (i, sub)
            seed_pair_cos[seed_pair] = cosines[i][sub]

    seed_pair_list = sorted(list(seed_pair_list))
    seed_pair_list_uni = sorted(list(seed_pair_list_uni))

    return seed_pair_list, seed_pair_list_uni, seed_pair_cos


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for seed_pair in seed_pair_list:
        rep, ent_id = seed_pair
        if ent_id not in pair_dict:
            if rep not in pair_dict:
                pair_dict.update({ent_id: rep})
            else:
                new_rep = pair_dict[rep]
                j = 0
                while rep in pair_dict:
                    new_rep = pair_dict[rep]
                    rep = new_rep
                    j += 1

                pair_dict.update({ent_id: new_rep})
        else:
            if rep not in pair_dict:
                new_rep = pair_dict[ent_id]
                if rep > new_rep:
                    pair_dict.update({rep: new_rep})
                else:
                    pair_dict.update({new_rep: rep})
            else:
                old_rep = rep
                new_rep = pair_dict[rep]
                j = 0
                while rep in pair_dict:
                    new_rep = pair_dict[rep]
                    rep = new_rep
                    j += 1

                if old_rep > new_rep:
                    pair_dict.update({ent_id: new_rep})
                else:
                    pair_dict.update({ent_id: old_rep})

    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        cluster_list[ent_id] = rep
    return cluster_list


class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, snapshot_side_infos, true_ent2clust, true_clust2ent, sub_uni2triple_dict=None):
        self.p = params
        self.snapshot_side_infos = snapshot_side_infos
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.sub2embed = {}
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.sub_uni2triple_dict = sub_uni2triple_dict


    def fit(self):

        show_memory = False
        if show_memory:
            print('show_memory:', show_memory)
            import tracemalloc
            tracemalloc.start(25)  # 默认25个片段，这个本质还是多次采样


        folder_to_make = '../output/' + self.p.dataset

        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)


        fname = folder_to_make + '/semantic_embedding'
        bert_model = Bert_Model(self.p, self.snapshot_side_infos, self.true_ent2clust,self.true_clust2ent)
        semantic_embeddings = bert_model.encode()
        pickle.dump(semantic_embeddings, open(fname, 'wb'))



