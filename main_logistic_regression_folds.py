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
from sklearn.linear_model import LogisticRegression
numIt = 100
gradIt = 20

def run_save(*args):    
    (s,cv,b,data_file_path)=args
    # sys.stdout = open(time.strftime("%d%m")+ '_%0.2g_%d_%d' %(s) + ".out", "w")
    np.random.seed()

    print("loading input data...")
    ip_pkl = np.load(data_file_path)
    X = ip_pkl['Xtrain']; comorbidities = ip_pkl['comorbidities_first_train']; Ainit= ip_pkl['Ainit']

    print("loading complete...")
    if not X.shape[0]==comorbidities.shape[0]:
        print("Error in input files!")
        sys.exit(2)
    print("no patients:", X.shape[0], " no of features:", X.shape[1], " no of conditions:", comorbidities.shape[1])
    
    A=np.zeros((X.shape[1], comorbidities.shape[1]))   
    W=np.zeros((X.shape[0], comorbidities.shape[1]))
    rk = np.shape(comorbidities)[1]
    print("Rank:%d, cv%d, s:%d, fit_bias:%d, rand:%f, xdensity:%f"  %(rk,cv,s*1000,b,np.random.rand(),X.data.size))
    model={}
    for j in range(rk):
        model[j]=LogisticRegression(penalty='l1', C=s, fit_intercept=b, class_weight='balanced')
        model[j].fit(X,comorbidities[:,j])
        #print model.classes_
        A[:,j]=model[j].coef_.ravel()
        W[:,j]=model[j].predict_proba(X)[:,1].ravel()
        
    r={'A':A,'W':W,'lossValuefinal':None,'stats':None,'fit_time':None,'model':model}
    filename='ad_NMF_cv%d_logistic_rk%d_s%d_b%d.pickle' %(cv,rk,s*1000,b)
    pickle.dump(r,open(filename,'wb'))
    print("Saved result pickle %f" %s)

    
def main1(argv):        
    b=1
    try:
        opts,args=getopt.getopt(argv,"hs:b:f:",["regularization=","bias=","data_file_path="])
    except getopt.GetoptError:
        print('main_logistic_regression.py -s <regularization> -b <bias 0/1> -f <data_file_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print('main_logistic_regression.py -s <regularization> -b <bias 0/1> -f <data_file_path>')
            sys.exit(0)
        elif opt in ('-s','--regularization'):
            ss=[float(a) for a in arg.split(',')]
        elif opt in ('-b','--bias'):
            b=int(arg)
        elif opt in ('-f','--data_file_path'):
            data_file_path = str(arg)
             
    jobs=[];    
    filenames=[data_file_path+'/EstimatorInput_strat_mortality_48_%d.pkl' %cv for cv in range(5)]
    for s in ss:
        for cv in range(5):
            p=multiprocessing.Process(target=run_save,args=(s,cv,b,filenames[cv]))
            jobs.append(p)
            p.start()
        
    for p in jobs:
        p.join()

if __name__ == '__main__':
    main1(sys.argv[1:])

