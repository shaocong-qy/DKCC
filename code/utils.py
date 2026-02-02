import random
import datetime
import numpy as np
import itertools
from collections import defaultdict
class EarlyStopping:
    def __init__(self, patience, delta, trace_func=print, type='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path1 = path1
        # self.path2 = path2
        self.trace_func = trace_func
        self.type = type

    def __call__(self, loss):

        if self.type == 'loss':
            score = -loss

            if self.best_score is None:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
                self.counter = 0

        elif self.type == 'acc':
            score = -loss

            if self.best_score is None:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
            elif score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
                self.counter = 0

    # def save_checkpoint(self, loss, model, model2):
    #     # if self.local_rank == 0:
    #     #     torch.save(model.module.state_dict(), self.path)
    #     torch.save(model.state_dict(), self.path1)
    #     torch.save(model2.state_dict(), self.path2)
    #
    #     self.val_loss_min = loss


def get_canseed_pair(ent_list, ent2id):
    seed_pair = []
    with open('data/myexp/originalcanseed.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            i=ent2id[res[0]]
            j=ent2id[res[1]]

            id_tuple = (i, j)
            seed_pair.append(id_tuple)
    return seed_pair

# 归一化
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# 余弦相似度
def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta


# 余弦距离
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_distance = 1 - cos_theta
    # cos_distance = 0.5 - 0.5 * cos_theta
    return cos_distance

def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

    return true_head, true_tail

def get_neg_samples(self, head, relation, tail,true_head, true_tail,mode='tail'):
    # true_head, true_tail = get_true_head_and_tail(self.trpIds)


    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < self.num_neg_objects:
        negative_sample = np.random.randint(len(self.ent_list), size=self.num_neg_objects * 2)

        if mode == 'head':
            mask = np.in1d(
                negative_sample,
                true_head[(relation, tail)],
                assume_unique=True,
                invert=True
            )
        elif mode == 'tail':
            mask = np.in1d(
                negative_sample,
                true_tail[(head, relation)],
                assume_unique=True,
                invert=True
            )
        else:
            raise ValueError('Training batch mode %s not supported' % mode)
        negative_sample = negative_sample[mask]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size

    return  np.concatenate(negative_sample_list)[:self.num_neg_objects]

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

def totol_cluster2pair(cluster_list):
    seed_pair_list, id_list = [], []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        if id not in id_list:
            id_list.append(id)
            index_list = [i for i, x in enumerate(cluster_list) if x == id]
            if len(index_list) > 1:
                iter_list = list(itertools.combinations(index_list, 2))
                seed_pair_list += iter_list
    return seed_pair_list


# def filter_tuples(tuple_list):
#     return [t for t in tuple_list if all(num <= 2137 for num in t)]


def build_seed_pair_dict(seed_pairs):
    seed_dict = defaultdict(list)

    for pair in seed_pairs:
        a, b = pair
        seed_dict[a].append(pair)
        seed_dict[b].append(pair)

    return dict(seed_dict)

def cluster_tuples(tuples_list):
    parent = {}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u

    # Initialize parent for all numbers
    for t in tuples_list:
        for num in t:
            if num not in parent:
                parent[num] = num

    # Union numbers in the same tuple
    for t in tuples_list:
        if len(t) == 0:
            continue
        first_num = t[0]
        for num in t[1:]:
            union(first_num, num)

    # Group numbers by their root
    clusters = defaultdict(list)
    for num in parent:
        root = find(num)
        clusters[root].append(num)

    # Convert clusters to sorted tuples
    result = [tuple(sorted(cluster)) for cluster in clusters.values()]

    return result

def swap_and_deduplicate(tuples_list):
    unique_tuples = list()
    for tpl in tuples_list:
        reversed_tpl = tpl[::-1]
        unique_tuples.append(tpl)
        unique_tuples.append(reversed_tpl)
    # random.shuffle(unique_tuples)
    return unique_tuples

def split_tuples(tuples_list):
    result = []
    for tpl in tuples_list:
        if len(tpl) == 2:
            result.append(tpl)
        else:
            # 生成所有两两组合
            for combo in itertools.combinations(tpl, 2):
                result.append(combo)
    return result


#过滤主语以外的种子对
def filter_tuples(tuple_list,sub_list):
    return [t for t in tuple_list if all(num in sub_list for num in t)]

def sample_hard_negatives(clusters, hard_nums=7):
    # 建立元素到聚类的映射并检查重复
    element_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for num in cluster:
            if num in element_cluster:
                raise ValueError(f"数字 {num} 出现在多个聚类中")
            element_cluster[num] = cluster_idx

    # 预处理每个聚类的负样本池
    cluster_neg_pools = []
    num_clusters = len(clusters)
    for i in range(num_clusters):
        neg_pool = []
        for j in range(num_clusters):
            if j != i:
                neg_pool.extend(clusters[j])
        cluster_neg_pools.append(neg_pool)

    # 为每个元素生成k个负样本
    negatives = {}
    for num in element_cluster:
        cluster_idx = element_cluster[num]
        neg_pool = cluster_neg_pools[cluster_idx]

        if not neg_pool:
            raise ValueError(f"数字 {num} 所在的聚类没有可用的负样本")

        # 使用random.choices允许重复采样
        sampled = random.choices(neg_pool, k=hard_nums)
        negatives[num] = sampled

    return negatives
def pair2triples(seed_pair_list, ent_list, ent2id, id2ent, ent2triple_id_list, trpIds):
    seed_trpIds = []

    for seed_pair in seed_pair_list:
        i, j = seed_pair[0], seed_pair[1]
        if i < len(ent_list) and j < len(ent_list):
            ent1, ent2 = ent_list[i], ent_list[j]

            for ent in [ent1, ent2]:
                triple_list = ent2triple_id_list[ent]
                for triple_id in triple_list:
                    triple = trpIds[triple_id]
                    if str(id2ent[triple[0]]) == str(ent1):
                        trp = (ent2id[str(ent2)], triple[1], triple[2])
                        seed_trpIds.append(trp)

                    if str(id2ent[triple[0]]) == str(ent2):
                        trp = (ent2id[str(ent1)], triple[1], triple[2])
                        seed_trpIds.append(trp)

                    if str(id2ent[triple[2]]) == str(ent1):
                        trp = (triple[0], triple[1], ent2id[str(ent2)])
                        seed_trpIds.append(trp)\

                    if str(id2ent[triple[2]]) == str(ent2):
                        trp = (triple[0], triple[1], ent2id[str(ent1)])
                        seed_trpIds.append(trp)



    return seed_trpIds


def freeze_except(model, exception_names=None, verbose=True):

    # 处理输入参数
    if exception_names is None:
        exception_names = []
    elif isinstance(exception_names, str):
        exception_names = [exception_names]

    # 首先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 解冻例外层
    found_exceptions = set()
    for name, param in model.named_parameters():
        # 检查是否在例外列表中
        if any(except_name in name for except_name in exception_names):
            param.requires_grad = True
            found_exceptions.add(name.split('.')[0])  # 只取顶层模块名

    # 打印信息
    if verbose:
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)

        print(f"模型总参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"冻结参数: {frozen_count} 层")
        print(f"可训练参数: {trainable_count} 层")

        if found_exceptions:
            print(f"解冻层: {', '.join(found_exceptions)}")
        else:
            print("警告: 未找到指定的例外层，所有参数已被冻结")

    return model

def format_time(seconds: float) -> str:
    """将秒数转换为易读的时间格式"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} 秒"
    else:
        td = datetime.timedelta(seconds=seconds)
        return str(td)[:-3]  # 去除微秒部分