import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from helper import *
from preprocessing import SideInfo  # For processing data and side information
from train_embeddings_multi_view_opiec import Embeddings
from utils import *
import argparse, pickle, codecs
from collections import defaultdict as ddict

''' *************************************** DATASET PREPROCESSING **************************************** '''

class Incrementl_Model_Main(object):

    def __init__(self, args):
        self.p = args
        self.read_triples()

    def read_triples(self):
        fname = self.p.out_path + self.p.file_triples  # File for storing processed triples
        self.triples_list = []  # List of all triples in the dataset
        self.amb_ent = ddict(int)  # Contains ambiguous entities in the dataset
        self.amb_mentions = {}  # Contains all ambiguous mentions
        self.isAcronym = {}  # Contains all mentions which can be acronyms

        self.snapshot_0 = []
        self.snapshot_1 = []
        self.snapshot_2 = []
        self.snapshot_3 = []
        self.snapshot_4 = []

        print('loading dataset:', args.dataset)
        if args.dataset.split('-')[0] == 'opiec':


            self.snapshot_0 = pickle.load(open('../' + args.dataset +'/snapshot_0', 'rb'))
            self.snapshot_1 = pickle.load(open('../' + args.dataset +'/snapshot_1', 'rb'))
            self.snapshot_2 = pickle.load(open('../' + args.dataset +'/snapshot_2', 'rb'))
            self.snapshot_3 = pickle.load(open('../' + args.dataset +'/snapshot_3', 'rb'))
            self.snapshot_4 = pickle.load(open('../' + args.dataset +'/snapshot_4', 'rb'))

            self.true_ent2clust = []
            self.true_clust2ent = []

            ''' Ground truth clustering for snapshot_0 '''
            self.true_ent2clust_snapshot_0 = ddict(set)
            for trp in self.snapshot_0:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_0[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent_snapshot_0 = invertDic(self.true_ent2clust_snapshot_0, 'm2os')

            ''' Ground truth clustering for snapshot_1 '''
            self.true_ent2clust_snapshot_1 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 :
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_1[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent_snapshot_1 = invertDic(self.true_ent2clust_snapshot_1, 'm2os')

            ''' Ground truth clustering for snapshot_2 '''
            self.true_ent2clust_snapshot_2 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_2[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent_snapshot_2 = invertDic(self.true_ent2clust_snapshot_2, 'm2os')

            ''' Ground truth clustering for snapshot_3 '''
            self.true_ent2clust_snapshot_3 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_3[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent_snapshot_3 = invertDic(self.true_ent2clust_snapshot_3, 'm2os')

            ''' Ground truth clustering for snapshot_4 '''
            self.true_ent2clust_snapshot_4 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3 + self.snapshot_4:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_4[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent_snapshot_4 = invertDic(self.true_ent2clust_snapshot_4, 'm2os')


            self.true_ent2clust.append(self.true_ent2clust_snapshot_0)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_1)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_2)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_3)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_4)

            self.true_clust2ent.append(self.true_clust2ent_snapshot_0)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_1)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_2)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_3)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_4)

        else:
            self.snapshot_0 = pickle.load(open('../' + args.dataset +'/snapshot_0', 'rb'))
            self.snapshot_1 = pickle.load(open('../' + args.dataset +'/snapshot_1', 'rb'))
            self.snapshot_2 = pickle.load(open('../' + args.dataset +'/snapshot_2', 'rb'))
            self.snapshot_3 = pickle.load(open('../' + args.dataset +'/snapshot_3', 'rb'))
            self.snapshot_4 = pickle.load(open('../' + args.dataset +'/snapshot_4', 'rb'))

            self.true_ent2clust = []
            self.true_clust2ent = []


            ''' Ground truth clustering for snapshot_0 '''
            self.true_ent2clust_snapshot_0 = ddict(set)
            for trp in self.snapshot_0:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_0[sub_u].add(trp['true_sub_link'])
            self.true_clust2ent_snapshot_0 = invertDic(self.true_ent2clust_snapshot_0, 'm2os')

            ''' Ground truth clustering for snapshot_1 '''
            self.true_ent2clust_snapshot_1 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_1[sub_u].add(trp['true_sub_link'])
            self.true_clust2ent_snapshot_1 = invertDic(self.true_ent2clust_snapshot_1, 'm2os')

            ''' Ground truth clustering for snapshot_2 '''
            self.true_ent2clust_snapshot_2 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_2[sub_u].add(trp['true_sub_link'])
            self.true_clust2ent_snapshot_2 = invertDic(self.true_ent2clust_snapshot_2, 'm2os')

            ''' Ground truth clustering for snapshot_3 '''
            self.true_ent2clust_snapshot_3 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_3[sub_u].add(trp['true_sub_link'])
            self.true_clust2ent_snapshot_3 = invertDic(self.true_ent2clust_snapshot_3, 'm2os')

            ''' Ground truth clustering for snapshot_4 '''
            self.true_ent2clust_snapshot_4 = ddict(set)
            for trp in self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3 + self.snapshot_4:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust_snapshot_4[sub_u].add(trp['true_sub_link'])
            self.true_clust2ent_snapshot_4 = invertDic(self.true_ent2clust_snapshot_4, 'm2os')


            self.true_ent2clust.append(self.true_ent2clust_snapshot_0)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_1)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_2)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_3)
            self.true_ent2clust.append(self.true_ent2clust_snapshot_4)

            self.true_clust2ent.append(self.true_clust2ent_snapshot_0)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_1)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_2)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_3)
            self.true_clust2ent.append(self.true_clust2ent_snapshot_4)


    def get_sideInfo(self):
        fname1 = self.p.out_path + self.p.file_snapshot_0_sideinfo_pkl
        fname2 = self.p.out_path + self.p.file_snapshot_1_sideinfo_pkl
        fname3 = self.p.out_path + self.p.file_snapshot_2_sideinfo_pkl
        fname4 = self.p.out_path + self.p.file_snapshot_3_sideinfo_pkl
        fname5 = self.p.out_path + self.p.file_snapshot_4_sideinfo_pkl

        if not checkFile(fname5):
            self.snapshot_0_side_info = SideInfo(self.p, self.snapshot_0, self.snapshot_0, 'snapshot_0')
            self.snapshot_1_side_info = SideInfo(self.p, self.snapshot_1, self.snapshot_0 + self.snapshot_1, 'snapshot_1')
            self.snapshot_2_side_info = SideInfo(self.p, self.snapshot_2, self.snapshot_0 + self.snapshot_1 + self.snapshot_2 , 'snapshot_2')
            self.snapshot_3_side_info = SideInfo(self.p, self.snapshot_3, self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3 , 'snapshot_3')
            self.snapshot_4_side_info = SideInfo(self.p, self.snapshot_4, self.snapshot_0 + self.snapshot_1 + self.snapshot_2 + self.snapshot_3 + self.snapshot_4, 'snapshot_4')


            pickle.dump(self.snapshot_0_side_info, open(fname1, 'wb'))
            pickle.dump(self.snapshot_1_side_info, open(fname2, 'wb'))
            pickle.dump(self.snapshot_2_side_info, open(fname3, 'wb'))
            pickle.dump(self.snapshot_3_side_info, open(fname4, 'wb'))
            pickle.dump(self.snapshot_4_side_info, open(fname5, 'wb'))

        else:
            self.snapshot_0_side_info = pickle.load(open(fname1, 'rb'))
            self.snapshot_1_side_info = pickle.load(open(fname2, 'rb'))
            self.snapshot_2_side_info = pickle.load(open(fname3, 'rb'))
            self.snapshot_3_side_info = pickle.load(open(fname4, 'rb'))
            self.snapshot_4_side_info = pickle.load(open(fname5, 'rb'))


    def embedKG(self):

        snapshot_side_infos = []
        snapshot_side_infos.append(self.snapshot_0_side_info)
        snapshot_side_infos.append(self.snapshot_1_side_info)
        snapshot_side_infos.append(self.snapshot_2_side_info)
        snapshot_side_infos.append(self.snapshot_3_side_info)
        snapshot_side_infos.append(self.snapshot_4_side_info)


        embed = Embeddings(self.p, snapshot_side_infos, true_ent2clust=self.true_ent2clust,
                           true_clust2ent=self.true_clust2ent)
        embed.fit()



def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
    parser.add_argument('-data', dest='dataset', default='reverb-0.8', help='Dataset to run CESI on')
    parser.add_argument('-split', dest='split', default='test', help='Dataset split for evaluation')
    parser.add_argument('-data_dir', dest='data_dir', default='../data', help='Data directory')
    parser.add_argument('-out_dir', dest='out_dir', default='../output', help='Directory to store CESI output')
    parser.add_argument('-reset', dest="reset", action='store_true', default=True,
                        help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-name', dest='name', default=None, help='Assign a name to the run')
    parser.add_argument('--use_assume', default=True)

    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--m', default=15, type=int)
    parser.add_argument('--hard_negs', default=7, type=int)
    parser.add_argument('--lambda_reg', default=0.1, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)



    parser.add_argument('--use_BERT', default=True)
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    # parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=False)
    parser.add_argument('--save_path', default='../output', type=str)

    # parser.add_argument('-d', '--hidden_dim', default=300, type=int)
    # parser.add_argument('-embed_dims', dest='embed_dims', default=300, type=int, help='Embedding dimension')

    parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'average'],
                        help='HAC linkage criterion')
    parser.add_argument('-metric', dest='metric', default='cosine',
                        help='Metric for calculating distance between embeddings')
    parser.add_argument('-num_canopy', dest='num_canopy', default=1, type=int,
                        help='Number of caponies while clustering')
    parser.add_argument('-true_seed_num', dest='true_seed_num', default=2361, type=int)
    parser.add_argument('-seed', default=2026,type=int)



    args = parser.parse_args()

    set_seed(args.seed)


    if args.name == None: args.name = args.dataset + '_' + args.split + '_' + '1'

    args.file_triples = '/triples.txt'  # Location for caching triples
    args.file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
    args.file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
    args.file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
    args.file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
    args.file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
    args.file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
    args.file_snapshot_0_sideinfo_pkl = '/snapshot_0_side_info.pkl'
    args.file_snapshot_1_sideinfo_pkl = '/snapshot_1_side_info.pkl'
    args.file_snapshot_2_sideinfo_pkl = '/snapshot_2_side_info.pkl'
    args.file_snapshot_3_sideinfo_pkl = '/snapshot_3_side_info.pkl'
    args.file_snapshot_4_sideinfo_pkl = '/snapshot_4_side_info.pkl'
    args.file_results = '/results.json'  # Location for loading hyperparameters

    args.out_path = args.out_dir + '/' + args.name  # Directory for storing output
    print('args.out_path:', args.out_path)
    print('args.reset:', args.reset)
    args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
    if args.reset: os.system('rm -r {}'.format(args.out_path))  # Clear cached files if requeste
    if not os.path.isdir(args.out_path): os.system(
        'mkdir -p ' + args.out_path)  # Create the output directory if doesn't exist

    CL_Model = Incrementl_Model_Main(args)  # Loading KG triples
    CL_Model.get_sideInfo()  # Side Information Acquisition
    CL_Model.embedKG()  # Learning embedding for Noun and relation phrases