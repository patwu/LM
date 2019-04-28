import argparse
import json
import numpy as np
import os
import sys
import pickle
import copy

from lstm import LSTMLM


def nearest_neighbor():
    lstm=LSTMLM(args)
    lstm.build_model()
    lstm.load_model('model')
    chars=pickle.load(open('chars.pkl','rb'))

    rchars={chars[c]:c for c in chars}
    while len(rchars)<args.vocab_size:
        rchars[len(rchars)]='<ERR>'
    
    vectors=lstm.get_vector()
    
    dists=[]

    for i in range(6000):
        dis=np.sum((vectors[i]-vectors[args.word_id])**2)
        dists.append((dis,rchars[i]))
    dists=sorted(dists,key=lambda tup:tup[0])
    for i in range(20):
        print dists[i][0],dists[i][1]

if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus',type=str,default='corpus')
    argparser.add_argument('--prefix_file', type=str,default='prefix.txt')

    argparser.add_argument('--max_length',type=int,default=64)

    argparser.add_argument('--n_emb',type=int,default=32)
    argparser.add_argument('--vocab_size', type=int,default=7000)
    argparser.add_argument('--n_hidden', type=int,default=128)
    argparser.add_argument('--batch_size', type=int,default=64)
    argparser.add_argument('--word_id', type=int, default=100)

    args = argparser.parse_args()
    print args

    nearest_neighbor()
