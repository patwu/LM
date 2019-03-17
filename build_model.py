import sys 
import argparse
import json
import numpy as np

def build_model():
    chars={'<BOS>':0,'<EOS>':1}
    sentences=[]
    with open(args.corpus,'r') as f:
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
    print len(chars)
    c_matrix=np.zeros((4000,4000))
    p_matrix=np.zeros((4000,4000))

    #build language model
    for s in sentences:
        segments=[c for c in s]
        segments.insert(0,'<BOS>')
        segments.append('<EOS>')
        for i in range(len(segments)-1):
            c1,c2=segments[i],segments[i+1]
            i1=chars[c1]
            i2=chars[c2]
            #count c matrix
            c_matrix[i1,i2]+=1
    #build p matrix
    for i in range(4000):
        sum_=np.sum(c_matrix[i,:])
        p_matrix[i]=c_matrix[i]/sum_
   
    return p_matrix,chars

def test(model):
    p_matrix,chars=model
    with open(args.testfile,'r') as f:
        lines=f.readlines()
        for line in lines:
            eprob=0.
            s=line.strip().decode('utf-8')
            segments=[c for c in s]
            segments.insert(0,'<BOS>')
            segments.append('<EOS>')
            str_=''
            for s in segments:
                str_+=s
            print len(segments),str_
            for i in range(len(segments)-1):
                c1,c2=segments[i],segments[i+1]
                i1=chars[c1]
                i2=chars[c2]
                #calc prob
                eprob+=np.log(p_matrix[i1][i2])
                print c1,c2,p_matrix[i1][i2]

            print line,eprob

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
                while True:
                    i2=np.random.choice(range(4000),p=p_mat[i1])
                    if p_mat[i1,i2]>0.01:
                        break;
                c2=rchars[i2]
                str_+=c2
                i1=i2
            print prefix,str_

if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus',type=str,default='corpus/wiki_00')
    argparser.add_argument('--testfile',type=str,default='test.txt')
    argparser.add_argument('--prefix_file', type=str,default='prefix.txt')
    args = argparser.parse_args()
    print args

    model=build_model()
    #test(model)
    maximun_generate(model)
