import argparse
import json
import numpy as np
import os
import sys
import pickle
import copy

from lstm import LSTMLM

def segmentation(s):
    segments=[c for c in s]
    segments.insert(0,'<BOS>')
    segments.append('<EOS>')
    return segments

def build_model():
    lstm=LSTMLM(args)
    lstm.build_model()

    chars={'<BOS>':0,'<EOS>':1,'<UNK>':2}
    sentences=[]
    sids=[]
    files=os.listdir(args.corpus)
    print files
    for file_ in files:
        al=os.path.join(args.corpus,file_)
        print al
        with open(al,'r') as f:
            lines=f.readlines()
            for line in lines:
                data=json.loads(line)
                text=data['text']
                sen=text.split('\n')
                for s in sen:
                    if len(s.strip())>0:
                        sentences.append(s)
                        sid=[0]
                        for c in s:
                            if not c in chars:
                                chars[c]=len(chars)
                            sid.append(chars[c])
                        sid.append(1)
                        sids.append(sid)
    n_char=len(chars)
    print 'vocabulary_size=%d'%n_char
    rchars={chars[c]:c for c in chars}

    for i in range(5000):
        batch_sen=np.random.choice(sids,size=args.batch_size)
        batch_sen=[copy.copy(s) for s in batch_sen]
        loss=lstm.train(batch_sen)
        if i%10==0:
            print i,loss    
    lstm.save_model('model')
    pickle.dump(chars,open('chars.pkl','wb'))
    return lstm,chars

def maximun_generate():
    lstm=LSTMLM(args)
    lstm.build_model()
    lstm.load_model('model')
    chars=pickle.load(open('chars.pkl','rb'))

    rchars={chars[c]:c for c in chars}
    while len(rchars)<args.vocab_size:
        rchars[len(rchars)]='<ERR>'

    with open(args.prefix_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            eprob=0.
            s=line.strip().decode('utf-8')
            segments=[c for c in s]
            segments.insert(0,'<BOS>')
            str_=''
            for s in segments:
                str_+=s
            prefix=str_
            
            ids=[chars[s] for s in segments]
            while len(ids)<args.max_length and ids[-1]!=1:
                dist=lstm.next_char([copy.copy(ids)])[0]
                i2=np.random.choice(range(len(rchars)),p=dist)
                eprob+=np.log(dist[i2])
                ids.append(i2)
                str_+=rchars[i2]
            print prefix,eprob,str_

if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus',type=str,default='corpus')
    argparser.add_argument('--prefix_file', type=str,default='prefix.txt')
    argparser.add_argument('--test_file',type=str,default='wiki_10')
    argparser.add_argument('--beam_width',type=int,default=5)

    argparser.add_argument('--max_length',type=int,default=64)

    argparser.add_argument('--n_emb',type=int,default=32)
    argparser.add_argument('--vocab_size', type=int,default=7000)
    argparser.add_argument('--n_hidden', type=int,default=128)
    argparser.add_argument('--batch_size', type=int,default=64)

    args = argparser.parse_args()
    print args

    #model=build_model()
    #test(model)
    maximun_generate()
    #beam_generate(model)
