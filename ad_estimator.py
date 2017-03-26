import numpy as np
import cPickle as pickle
import seaborn as sns
import cnmf
import time

tau=0.1 # line search parameter

class ad_estimator:
    # X  = WA^T    
    A=None
    W=None
    stats={};
    def __init__(self, X, R, projW, projA, Dcol, bias=0, numIt=50, gradIt=10, debug=1, verbose=0, func_kwargs={}):
        ''' 
        Inputs:
        X: EHR matrix to be factorized as WA.T
        R: rank of the factorization
        projW, projA: functions for projecting factors W and A respectively. 
        func_args={'function_name':{args for function}} arguments for D, projA, projW, D
        numIt: number of outer iterations
        gradIt: number of inner iterations
        
        Usage:
        create a member object
        call initialize
        call fit
        '''
        
        self.X = X        
        [self.N,self.p] = X.shape
        self.R = R        
        self.projW=projW
        self.projA=projA
        self.Dcol=Dcol
        self.numIt = numIt
        self.gradIt = gradIt
        self.bias = bias
        self.debug = debug
        self.verbose = verbose
        self.func_kwargs =func_kwargs       
        
    def initialize(self, A=None,W=None):
        print "Initializing ad_nmf object"
        R=self.R+self.bias
        if (W is None):
            self.W=np.random.rand(self.N,R)
        else:
            assert (W.shape[0]==self.N and W.shape[1]==R), "dimensions of W expected to be = %d, %d" % (self.N, R)
            self.W=W
            
        if (A is None):
            self.A=np.random.rand(self.p,R)
        else:
            assert (A.shape[0]==self.p and A.shape[1]==R), "dimensions of A expected to be = %d, %d" % (self.p, R)
            self.A=A

    def fit(self,filename=None, parallel=0):
        t=time.time()
        print("verbose in fit", self.verbose)
        if ((self.A is None) or (self.W is None)):
            self.intialize(self.A,self.W)
        if parallel:
            self.W,self.A,self.fval,self.stats=cnmf.constrained_nmf_par(X=self.X, projW=self.projW,\
                                                                        projA=self.projA,Dcol=self.Dcol,W=self.W, A=self.A,\
                                                                        bias=self.bias,numIt = self.numIt, gradIt=self.gradIt,\
                                                                        debug=self.debug, verbose=self.verbose,\
                                                                        func_kwargs=self.func_kwargs)
        else:           
            self.W,self.A,self.fval,self.stats=cnmf.constrained_nmf_serial(X=self.X, projW=self.projW,\
                                                                           projA=self.projA,Dcol=self.Dcol,W=self.W, A=self.A,\
                                                                           bias=self.bias,numIt = self.numIt, gradIt=self.gradIt,\
                                                                           debug=self.debug, verbose=self.verbose,\
                                                                           func_kwargs=self.func_kwargs)
        self.fit_time=time.time()-t  
        if self.debug:
            self.plotStats(filename)
    
        
    def normalizeFactors(self):
        print("TODO: add post-hoc normalization for interpretation; Currently done in pot files")
    
    def save(self,filename):
        print("Saving results, total run time:", self.fit_time)
        r={'A':self.A,'W':self.W,'lossValuefinal':self.fval,'stats':self.stats,'fit_time':self.fit_time}
        pickle.dump(r,open(filename,'wb'))
    
    def plotStats(self,filename=None):
        sns.plt.switch_backend('agg')
        sns.plt.subplot(1, 2, 1); sns.plt.plot(self.stats['fiter'])
        sns.plt.subplot(1, 2, 2); sns.plt.plot(self.stats['Median_nnz'])
        if not(filename==None):
            sns.plt.savefig(filename)
        else:
            filename = 'stats_last_run.png'
            sns.plt.savefig(filename)
            print("Saving plots to ", filename)
