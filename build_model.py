import sys 
import argparse
import json
import numpy as np

def build_model():
    chars={}
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
        for i in range(len(s)-1):
            c1,c2=s[i],s[i+1]
            i1=chars[c1]
            i2=chars[c2]
            #count c matrix
            c_matrix[i1][i2]+=1
    
    for i in range(4000):
        sum_=np.sum(c_matrix[i,:])
        p_matrix[i]=c_matrix[i]/sum_
   
    return p_matrix,chars

def test(model):
    p_matrix,chars=model
    with open(args.testfile,'r') as f:
        lines=f.readlines()
        prob=1.
        for line in lines:
            s=line.strip().decode('utf-8')
            print len(s),s
            for i in range(len(s)-1):
                c1,c2=s[i],s[i+1]
                i1=chars[c1]
                i2=chars[c2]
                prob*=p_matrix[i1][i2]
                print c1,c2,p_matrix[i1][i2]

            print line,prob


if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus',type=str,default='corpus/wiki_00')
    argparser.add_argument('--testfile',type=str,default='test.txt')
    args = argparser.parse_args()
    print args

    model=build_model()
    test(model)
