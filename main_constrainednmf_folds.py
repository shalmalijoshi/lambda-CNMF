# -*- coding: utf-8 -*-
"""
@author: shalmali
"""
import os
import numpy as np
import cPickle as pickle
import time
import multiprocessing
import ad_estimator as ad_nmf
from utils import D
from ad_nmf_helper import *
import sys, getopt

numIt = 100
gradIt = 20

def run_save(*args):
    (s,tau,i,loss,b,data_file,result_file,parallel,cv)=args
    sys.stdout = open(time.strftime("%d%m")+ '_cv%d_s%0.2g_t%d_i%d' %(cv,s,int(tau*1000),i) + ".out", "w")
    np.random.seed()

    print("loading input data...")
    ip_pkl = np.load(data_file)
    X = ip_pkl['Xtrain']; comorbidities = ip_pkl['comorbidities_first_train']; Ainit= ip_pkl['Ainit']


    print("loading complete...")
    if not X.shape[0]==comorbidities.shape[0]:
        print("Error in input files!")
        sys.exit(2)
    print("no patients:", X.shape[0], " no of features:", X.shape[1], " no of conditions:", comorbidities.shape[1])
    if (i==0 and not(Ainit is None)):
        print("Initializing phenotype matrix with default from :", data_file_path)
    else:
        print("Random initialization of phenotype matrix")
        Ainit = np.random.rand(X.shape[1], comorbidities.shape[1])

    if (s>0):
        Ainit = s*(Ainit/Ainit.sum(axis=0))

    rk = np.shape(comorbidities)[1]
    print("Rank:%d, cv:%d, s:%d, tau:%s, fit_bias:%d, run_id:%d, rand:%f, loss:%s, xdensity:%f"  %(rk,cv,s*100,str(tau),b,i,np.random.rand(),loss, X.data.size))

    if (parallel):
        loss = loss + '_par'
    Dcol = D[loss]

    if (s>0):
        print("simplex projection, support constraints")
        #if b:
        #    supportW=np.column_stack((comorbidities>0,w_bias*np.ones(comorbidities.shape[0],dtype=bool)))
        func_kwargs={'projA':{'simplex':s,'bias':b},\
                     'projW':{'support':comorbidities>0,'lb':tau,'ub':1.0,'bias':b}}

        ad_nmf_obj = ad_nmf.ad_estimator(X=X, R=rk, projW=projWsupport, projA=projAsimplex, Dcol=Dcol, bias=b, numIt=numIt, gradIt=gradIt,verbose=1,func_kwargs=func_kwargs)

        if b: ad_nmf_obj.initialize(A=np.hstack((Ainit,0.1*np.random.rand(Ainit.shape[0],1))),\
                                    W=np.hstack((comorbidities,0*np.array(X.mean(1)))))
        else: ad_nmf_obj.initialize(A=Ainit,W=comorbidities)
        ad_nmf_obj.fit('stats_cv%d_%s_rk%d_s%d_i%d_p%d_b%d_t%d.png' %(cv,loss,rk,s*100,i,parallel,b,tau*1000), parallel)
        print("Finished fitting")

    elif (s==0):
        print("no simplex projection, support constraints")
        func_kwargs={'projA':{'lb':0.0,'bias':b},\
                     'projW':{'support':comorbidities>0,'lb':tau,'ub':1.0,'bias':b}}

        ad_nmf_obj = ad_nmf.ad_estimator(X=X, R=rk, projW=projWsupport, projA=projBound, Dcol=Dcol, bias=b, numIt=numIt, gradIt=gradIt,verbose=1,func_kwargs=func_kwargs)

        if b: ad_nmf_obj.initialize(A=np.hstack((Ainit,0.1*np.random.rand(Ainit.shape[0],1))),\
                                    W=np.hstack((comorbidities,0*np.array(X.mean(1)))))
        else: ad_nmf_obj.initialize(A=Ainit,W=comorbidities)
        ad_nmf_obj.fit('stats_cv%d_%s_rk%d_s%d_i%d_p%d_b%d_t%d.png' %(cv,loss,rk,s*100,i,parallel,b,tau*1000), parallel)
        print("Finished fitting")

    else:
        print("Invalid s argument: valid usage s>=0")
        sys.exit(2)

    ad_nmf_obj.save(result_file)
    #ad_nmf_obj.save('ad_NMF_cv%d_%s_rk%d_s%d_i%d_p%d_b%d_t%d.pickle' %(cv,loss,rk,s*100,i,parallel,b,tau*1000))
    print("Saved result pickle")


def main1(argv):
    loss = 'sparse_poisson'
    b=1
    parallel=0
    ids=range(6)
    tau=0.01
    ids=[1]
    s=[1.0]
    data_file_path='./'
    result_file='result.pickle'
    try:
        opts,args=getopt.getopt(argv,"hs:l:i:b:f:t:p:",["simplexs=","loss=","run_ids=","bias=","data_file_path=","tau=",'parallel='])
    except getopt.GetoptError:
        print('main_constrainednmf.py -s <simplex constraint values> -t <tau> -l <loss>  -i <run_ids> -b <bias 0/1> -p <parallel optimization 0/1> -f data_file_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print('main_constrainednmf.py -s <simplex constraint values> -t <tau> -l <loss>  -i <run_ids> -b <bias 0/1> -p <parallel optimization 0/1> -f data_file_path>')
            sys.exit(0)
        elif opt in ('-s','--simplexs'):
            # Comma separated simplex contraints value (lambda parameter in the paper). Default 1.0
            ss=[float(a) for a in arg.split(',')]
        elif opt in ('-i','--run_ids'):
            # Comma separated run_ids (multiple initializations). Default 1
            ids=[int(a) for a in arg.split(',')]
        elif opt in ('-l','--loss'):
            # Loss value: currently supports sparse_gaussian and sparse_poisson. Default sparse_poisson
            loss=str(arg)
        elif opt in ('-b','--bias'):
            # Include feature bias/not, no patient bias in the current code, change w_bias in cnmf.py to 1 for patient bias. Default 1
            b=int(arg)
        elif opt in ('-p','--parallel'):
            # Use parallelization in optimization of each factor (preliminary code not fully tested). Default 0.
            b=int(arg)
        elif opt in ('-f','--data_file_path'):
            # Root director where EstimatorInput_strat_<cv>.pkl files are present. Default ./
            data_file_path = str(arg)
        elif opt in ('-r','--result_file'):
            # File name for storing the results. Default 'result.pickle'
            result_file = str(arg)
        elif opt in ('-t','--tau'):
            # Tau parameter in the paper. Default 0.01
            tau = float(arg)

    jobs=[];
    filenames=[data_file_path+'/EstimatorInput_strat_mortality_48_%d.pkl' %cv for cv in range(5)]
    for i in ids:
        for s in ss:
            for cv in range(5):
                p=multiprocessing.Process(target=run_save,args=(s,tau,i,loss,b,filenames[cv],result_file,parallel,cv))
                jobs.append(p)
                p.start()

    for p in jobs:
        p.join()

if __name__ == '__main__':
    main1(sys.argv[1:])
