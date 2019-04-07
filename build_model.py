import sys 
import argparse
import json
import numpy as np
import os

def segmentation(s):
    segments=[c for c in s]
    segments.insert(0,'<BOS>')
    segments.append('<EOS>')
    return segments

def build_model():
    chars={'<BOS>':0,'<EOS>':1,'<UNK>':2}
    sentences=[]
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
                        for c in s:
                            if not c in chars:
                                chars[c]=len(chars)
    n_char=len(chars)
    print n_char
    c_matrix=np.zeros((n_char,n_char))
    p_matrix=np.zeros((n_char,n_char))

    #build language model
    for s in sentences[:1000]:
        segments=segmentation(s)
        for i in range(len(segments)-1):
            c1,c2=segments[i],segments[i+1]
            i1=chars[c1]
            i2=chars[c2]
            #count c matrix
            c_matrix[i1,i2]+=1
    c_matrix+=0.001
    #build p matrix
    for i in range(n_char):
        sum_=np.sum(c_matrix[i,:])
        p_matrix[i]=c_matrix[i]/sum_
   
    return p_matrix,chars

def test(model):
    p_matrix,chars=model
    total_pp=0.
    n_sentence=0
    with open(args.test_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            data=json.loads(line)
            text=data['text']
            sen=text.split('\n')
            for s in sen:
                if len(s.strip())>0:
                    eprob=0.
                    segments=segmentation(s)
                    str_=''
                    for s in segments:
                        str_+=s
                    for i in range(len(segments)-1):
                        c1,c2=segments[i],segments[i+1]
                        if c1 in chars:
                            i1=chars[c1]
                        else:
                            i1=2 #UNK
                        if c2 in chars:
                            i2=chars[c2]
                        else:
                            i2=2 #UNK
                        #calc prob
                        eprob+=np.log(p_matrix[i1][i2])
                    epp=-eprob/(len(segments)-1)
                    print epp,eprob,str_
                    n_sentence+=1
                    total_pp+=(epp-total_pp)/n_sentence
    print n_sentence,total_pp


def maximun_generate(model):
    p_mat,chars=model
    rchars={chars[c]:c for c in chars}
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
            i1=chars[segments[-1]]
            while i1!=1 and len(str_)<50:
                i2=np.random.choice(range(len(chars)),p=p_mat[i1])
                c2=rchars[i2]
                str_+=c2
                i1=i2
            print prefix,str_

def beam_generate(model):
    p_mat,chars=model
    rchars={chars[c]:c for c in chars}
    with open(args.prefix_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            s=line.strip().decode('utf-8')
            segments=[c for c in s]
            segments.insert(0,'<BOS>')
            prefix=''.join(segments)
            i1=chars[segments[-1]]
            beam=[(i1,prefix,0.)]
            result=[]
            while len(result)<10:
                nbeam=[]
                for b in beam:
                    i1,str_,eprob=b;
                    i2_list=np.random.choice(a=range(len(chars)),size=args.beam_width,p=p_mat[i1],replace=False)
                    for i2 in i2_list:
                        c2=rchars[i2]
                        nstr_=str_+c2
                        neprob=eprob+np.log(p_mat[i1,i2])
                        if i2==1 or len(nstr_)>50:
                            result.append((nstr_,neprob))
                        else:
                            nbeam.append((i2,nstr_,neprob))
                nbeam=sorted(nbeam,key=lambda be:be[2],reverse=True)
                beam=nbeam[:args.beam_width]
            for str_,eprob in result:
                print eprob,str_
if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus',type=str,default='corpus')
    argparser.add_argument('--prefix_file', type=str,default='prefix.txt')
    argparser.add_argument('--test_file',type=str,default='wiki_10')
    argparser.add_argument('--beam_width',type=int,default=5)
    args = argparser.parse_args()
    print args

    model=build_model()
    #test(model)
    #maximun_generate(model)
    beam_generate(model)
