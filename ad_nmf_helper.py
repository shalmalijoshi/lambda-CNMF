"""
@author: shalmali
"""
import numpy as np
from sys import exit
from utils import euclidean_proj_simplex,euclidean_proj_nnl1ball
from numba import jit
from multiprocessing import Pool, Array, Value
#from joblib import Parallel, delayed

minb=0
@jit(nogil=True,cache=True)
def projBound(X, lb=0, ub=None, bias=1):
    Xc = np.empty(X.shape)
    if bias:
        Xc[:,:-1] = np.clip(X[:,:-1],lb,ub)
        Xc[:,-1] = np.clip(X[:,-1],minb,None)
    else:
        Xc = np.clip(X,lb,ub)
    return Xc

@jit(nogil=True,cache=True)
def projAsimplex(A,simplex=1.0,bias=1):
    ''' A is p x r matrix. 
    projects columns of A onto a scaled simplex of scale "simplex"'''
    Aproj=np.empty(A.shape)
    if bias:
        for j in range(A.shape[1]-1):
            Aproj[:,j]=euclidean_proj_simplex(A[:,j],simplex)
        Aproj[:,-1] = np.clip(A[:,-1],minb,None)
    else:   
        for j in range(A.shape[1]):
            Aproj[:,j]=euclidean_proj_simplex(A[:,j],simplex)
    return Aproj

@jit(nogil=True,cache=True)
def projAnnl1ball(A,s=1.0,bias=1):
    ''' A is p x r matrix. 
    projects columns of A onto a scaled simplex of scale "simplex"'''
    Aproj=np.empty(A.shape)
    if bias:
        for j in range(A.shape[1]-1):
            Aproj[:,j]=euclidean_proj_nnl1ball(A[:,j],s)
        Aproj[:,-1] = np.clip(A[:,-1],minb,None)
    else:   
        for j in range(A.shape[1]):
            Aproj[:,j]=euclidean_proj_nnl1ball(A[:,j],s)
    return Aproj

@jit(nogil=True,cache=True)
def projWsupport(W, support=None, lb=0.0, ub=1.0, bias=1, w_bias=0):
    ''' W,support are N x r matrices
    Projects rows of W onto the support set given by support
    and bounds given by lb and ub'''
    
    Wproj=np.clip(W,lb,ub)
    if bias:
        if (support is not None): 
            Wproj[:,:-1] = (Wproj[:,:-1]*support)        
        Wproj[:,-1] = np.clip(W[:,-1],w_bias*minb,None)
    else:
        if (support is not None): 
            Wproj = (W*support)      
    return np.abs(Wproj)

@jit(nogil=True,cache=True)
def projWsupport_l1(W, support=None, bias=0, w_bias=0):
    Wproj=np.zeros(W.shape)
    if W.shape[1]-support.shape[1]:
        support =np.hstack((support, w_bias*np.ones(W.shape[0],dtype=bool)))
        
    if (support is not None): 
        for j in range(W.shape[0]):
            Wproj[j,support[j,:]>0] = euclidean_proj_simplex(W[j,support[j,:]>0],1.0)     
    else:
        if bias:
            for j in range(W.shape[0]):
                Wproj[j,:(Wproj.shape[1]-1)] = euclidean_proj_simplex(W[j,:(Wproj.shape[1]-1)],1.0)
        else:
            for j in range(W.shape[0]):
                Wproj[j,:] = euclidean_proj_simplex(W[j,:],1.0)
 
    if bias: 
        Wproj[:,-1] = np.clip(W[:,-1],minb,None)
    return np.abs(Wproj)

def read_llda_results(fname):
    llda_train = []
    K=30+1
    with open(fname) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_arr = np.zeros(K)
            for i in range(1,len(row),2):
                row_arr[int(row[i])] = float(row[i+1])
            llda_train.append(row_arr)
    llda_train = np.array(llda_train)
    return llda_train


#X=UV.T+b where V is the variable factor U is the fixed factor (gradient computed wrt V.T columnwise)
def factorDivergenceFunction(V,g,U,X,b,Dcol,Dcol_kwargs, support=None):
    N=V.shape[0]
    f=0.0

    if g:
        gradV=np.zeros(V.shape)
    if support is None:
        for j in xrange(N):
            if not(Dcol_kwargs=={}):
                dout=Dcol(w=V[j,:],A=U,x=X.getcol(j),b=b,g=g,**Dcol_kwargs)
            else:
                dout=Dcol(w=V[j,:],A=U,x=X.getcol(j),b=b,g=g)
            if g:
                ff,gg=dout
                f=f+ff
                gradV[j,:]=gg
            else:
                f=f+dout
    else:         
        #Not verified
        if Dcol_kwargs: Asum=np.copy(Dcol_kwargs['A_sum'])
        for j in xrange(N):           
            if not(Dcol_kwargs=={}):
                Dcol_kwargs['A_sum']=Asum[support[j,:]]
                dout=Dcol(w=V[j,support[j,:]],A=U[:,support[j,:]],x=X.getcol(j),b=b,g=g,**Dcol_kwargs)                
            else:
                dout=Dcol(w=V[j,support[j,:]],A=U[:,support[j,:]],x=X.getcol(j),b=b,g=g)
            if g:
                ff,gg=dout
                f=f+ff
                gradV[j,support[j,:]]=gg
            else:
                f=f+dout
        if Dcol_kwargs: Dcol_kwargs['A_sum']=Asum
    if g:
        return f,gradV
    return f


        
#=======================
#@jit(nogil=True,cache=True)
#X=UV.T+b where V is the variable factor U is the fixed factor (gradient computed wrt V.T columnwise)
def parfactorDivergenceFunction(V,g,U,X,b,Dcol,Dcol_kwargs,p_pool):
    N=V.shape[0]
    f=0.0    
    args=[((V[j,:],U,X.getcol(j),b,g),Dcol_kwargs) for j in xrange(N)]   
    res = p_pool.map(Dcol,args)
    if g:
        gradV=np.zeros(V.shape)
        j=0
        for ff,x in res:
            f +=ff
            gradV[j,:]= x
            j+=1
        return f,gradV
    else:
        f = np.asarray(res)
        f = f.sum()
        return f
#=======================

# Converts to columnwise/rowwise factors for coordinate descent type algos
# This assumes the argument list is in a dictionary
def wrap_proj_args(projWargs,projAargs):
        if 'support' not in projWargs:
            projWargs['support'] = np.ones(W.shape)
            print('support constraints not provided, defaulting to full support on W')
        if 'simplex' not in projAargs:
            projAargs['simplex'] = 1
            print('simplex constraints nor provided, defaulting to the probability simplex!')
        projargs_tuples = ()
        projfuncs=()
        for n in range(W.shape[0]):
            projargs_tuples = projargs_tuples + (projWargs['support'][n,:],)
            projfuncs = projfuncs + (projW,)
        for r in range(A.shape[1]):
            projargs_tuples = projargs_tuples + (projAargs['simplex'],)
            projfuncs = projfuncs + (projA,)
        return projargs_tuples,projfuncs
        
# Converts to columnwise/rowwise factors for coordinate descent type algos
def mat_pair_to_tuples(W,A):
    flat_tuples = to_tuple(W)
    flat_tuples = flat_tuples + to_tuple(A.T)
    return flat_tuples

# This assumes the argument list is in a dictionary          
def nmfoptwrapper(W,A,projW=None,projA=None,projWargs=None,projAargs=None):
    factor_tuples = (W,A)
    if projW==None:
        projW = lambda x:x
        projWargs = []
    if projA==None:
        projA = lambda x:x
        projAargs=[]

    if projW==projWsupport and (projWargs==None or 'support' not in projWargs):
        projWargs = np.ones(W.shape)
    elif projW==projWsupport and 'support' in projWargs:
        projWargs = projWargs['support']
    #else:
    #    print "projection function:", projW, " not supported for W!"
    #    exit()
    if projA==projAsimplex and (projAargs==None or 'simplex' not in projAargs):
        projAargs = 1
    elif projA==projAsimplex and 'simplex' in projAargs:
        projAargs = projAargs['simplex']
    #else:
    #    print "projection function:", projA, " not supported for A!"
    #    exit()

    projargs = (projWargs,projAargs)
    projfuncs = (projW,projA)
    return factor_tuples,projfuncs,projargs

def to_tuple(a):
    try:
        return tuple(np.asarray(x) for x in a)
    except TypeError:
        return a
    
def optnmfwrapper(factor_tuples):
    # the first N tuples here go in to the W matrix and rest to the A matrix
    W = np.array(factor_tuples[0])
    #print "W = ",np.shape(W)
    #W = np.array(all_arr[range(N),:])
    A = np.array(factor_tuples[1])
    #print "A = ", np.shape(A)
    return W,A

def simulate():
    N = 500
    p = 100
    R=30
    A = np.random.rand(p,R)
    W = np.random.rand(N,R)
    X = sp.csc_matrix(W.dot(A.T))
    #support = np.ones([N,R])
    support = np.random.randint(2,size=(N,R))
    simplex = np.floor(np.median(np.sum(A,0)))# should give at least some sparsity
    print("simplex = ", simplex)
    loss = 'sparse_gaussian'
    adnmf_obj = ad_estimator(X,R,support,simplex,projWsupport,projAsimplex,loss,D,numIt=100)
    #adnmf_obj.initialize(A,W)
    adnmf_obj.initialize(2*A,5*W)
    return adnmf_obj

