"""
@author: suriya
"""

import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from numba import jit
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

class KeyboardInterruptError(Exception): pass

maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=0
class KeyboardInterruptError(Exception): pass

def min_prec_rec(ytest,ypred):
    prec, rec, thr = metrics.precision_recall_curve(ytest,ypred)
    diff = np.abs(prec-rec)
    idx = np.argmin(diff)
    #print idx,prec[idx],rec[idx],thr[idx]
    #print prec,rec
    score = np.min([prec[idx],rec[idx]])
    return score

def min_prec_rec_v2(ytest,ypred):
    prec, rec, thr = metrics.precision_recall_curve(ytest,ypred)
    scorevec = np.minimum(prec, rec)
    score = np.max(scorevec)
    return score

def fit_predict_classification(X,y,Xtest,ytest,scorer_func=min_prec_rec,penalty='l2',param={'C':[1000,100,10,1,0.1,0.01,0.001,0.0001]},scale=1):
    #Linear SVC with hinge loss
    metric="roc_auc"
    print "Shapes:", X.shape,y.shape,Xtest.shape,ytest.shape

    #param={'C':[1000,100,10,1,0.1,0.01,0.001,0.0001]}
    #param={'C':[0.1]}
    #param={'C':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]}
    if metric=="min_prec_recall":
        scorer = metrics.make_scorer(scorer_func,greater_is_better=True,needs_threshold=True)

        model=GridSearchCV(LogisticRegression(class_weight='balanced',penalty=penalty,intercept_scaling=scale),param,scoring=scorer)
    elif metric:
        model=GridSearchCV(LogisticRegression(class_weight='balanced',penalty=penalty,intercept_scaling=scale),param,scoring=metric)
    
    model.fit(X,y)
    print "Model:LinearSVC, metric:%s, best_param:" %(metric), model.best_params_
    print model.grid_scores_
    
    ypred=model.decision_function(Xtest)
    
    return {'ypred': ypred, 'ytest': ytest, 'auprc': metrics.average_precision_score(ytest,ypred), 'roc': metrics.roc_auc_score(ytest,ypred),'model': model.best_estimator_}
    #return {'ypred': ypred, 'ytest': ytest, 'auprc': metrics.precision_score(ytest,ypred)}

def fit_predict_classification_pr(X,y,Xtest,ytest):
    #Linear SVC with hinge loss
    metric="roc_auc_score"
    param={'C':[100,10,1,0.1,0.01,0.001,0.0001]}
    model=GridSearchCV(svm.LinearSVC(loss='hinge'),param,scoring=metric)
    model.fit(X,y)
    
    print "Model:LinearSVC, metric:%s, best_param:" %(metric), model.best_params_
    print model.grid_scores_
    
    ypred=model.decision_function(Xtest)
    
    return {'ypred': ypred, 'ytest': ytest, 'auprc': metrics.roc_auc_score(ytest,ypred)}

   

@jit(nogil=True,cache=True)
def euclidean_proj_nnl1ball(v, s=1):
    '''Uses Result from Tandon, Sra paper'''
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of non-negative values
    u = v.copy(); 
    u[u<0]=0;
    # check if u is already a solution
    if u.sum() <= s:
        return u    
    
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    return euclidean_proj_simplex(u, s=s)

@jit(nogil=True,cache=True)
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
    min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
    n-dimensional vector to project
    s: int, optional, default: 1,
    radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
    Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
    International Conference on Machine Learning (ICML 2008)
    http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if np.abs(v.sum()-s)<1e-15 and np.alltrue(v >= 0):
        # best projection: itself!
        return s*v/v.sum()
    
    #floating point error fixes
    if (np.abs(v.clip(min=0).sum()-s)<1e-15):
        w = v.clip(min=0)
        return s*w/w.sum()
    
    # get the array of cumulative sums of a sorted (decreasing) copy of v       
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution    
    if (len(np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0])>0):
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    else:
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        #print "Warning: IndexError in projection, s=%f" %s
        return w
    
    if (rho==0):
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        return w;        
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    #compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    if (w.sum()==0):
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        #print "Warning: w was set to zero, s=%f" %s
        return w
    
    #w=(s/np.sum(w))*w  
    assert np.alltrue(w>=0), np.where(w<0)
    
    return w

@jit(nogil=True,cache=True)
def D_spGauss(w,A,x,b,g=1,A_sum=None,b_sum=None):
    f = 0.0
    assert sp.isspmatrx_csc(x), 'D defined only for sparse csc X'
    
    xhat = A.dot(w)+b
    f = 0.5*(np.sum((xhat[x.indices]-x.data)**2)) + 0.5*(np.sum(xhat**2)-np.sum(xhat[x.indices]**2))
    if g:
        gradF = (A.T).dot(xhat)-((A[x.indices]).T).dot(x.data)
        return f,gradF
    else: return f

@jit(nogil=True,cache=True)
def D_spPoisson(w,A,x,b,g=1,A_sum=None,b_sum=None):
    f=0.0
    assert sp.isspmatrix_csc(x), 'D defined only for sparse X'

    xhat_ind=A[x.indices].dot(w)
    if (A_sum is None): A_sum=np.sum(A,0)
    if np.isscalar(b):
        xhat_ind = xhat_ind+b
        if b_sum is None: b_sum=b*A.shape[0]
        b_ind=b
    else:
        xhat_ind = xhat_ind+b[x.indices]
        if b_sum is None: b_sum=np.sum(b)
        b_ind=b[x.indices]
        
    if g:
        if (np.any(xhat_ind==0)):
            f = maxval                
            gradF = np.zeros(w.shape)
            weps = np.copy(w)
            for i in range(len(w)):
                #print 'in here print', np.shape(weps)
                weps[i] += eps
                #print 'in here'
                xhateps_ind = A[x.indices].dot(weps)+b_ind
                #print 'successfully added bias'
                if np.any(xhateps_ind==0):
                    feps = maxval
                else:
                    feps= A_sum.dot(weps)+b_sum-np.sum(x.data)+np.sum(x.data*np.log(x.data/xhateps_ind))
                gradF[i] = (feps-f)/eps
                weps[i] -= eps                    
        else:
            t = (x.data/xhat_ind)
            f = A_sum.dot(w)+b_sum-np.sum(x.data)+np.sum(x.data*np.log(t))
            f = min(f,maxval)
            gradF = A_sum-A[x.indices].T.dot(t)
        return f,gradF
    else:
        if (np.any(xhat_ind==0)):
            f = maxval
        else:
            t = (x.data/xhat_ind)
            f = A_sum.dot(w)+b_sum-np.sum(x.data)+np.sum(x.data*np.log(t))
            f = min(f,maxval)                
        return f

#@jit(nogil=True,cache=True)
def D_spPoissonpar(inargs):
    # print "input to loss:", inargs, "kwargs:", kwargs
    try:
        return D_spPoisson(*inargs[0],**inargs[1])
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


D={'sparse_gaussian':D_spGauss,'sparse_poisson':D_spPoisson, 'sparse_poisson_par': D_spPoissonpar}
#==================================            

def proj_support_bound(v,s_vec=None,lb = 0.0, ub = 1.0):
    #print "v:", v, len(v)
    #print "svec:", s_vec, len(s_vec)
    #print "assert:", len(v)==len(s_vec)
    # v: Input vector
    # s_vec: Binary vector imposing support constraints
    assert lb < ub, "Bound constraints improper!"
    if s_vec==None:
        s_vec = np.ones(v.shape)
    
    assert len(v)==len(s_vec), "input vector and support vector dimensions do not match!"
    assert np.all([(x==0 or x==1) for x in s_vec]), "Function expects s_vec to be a binary vector"
    #print "v before", v    
    v = np.multiply(v,s_vec) # element-wise multiplication
    #print "v after", v
    #print "svec",s_vec
    v[v < lb] = lb
    v[v > ub] = ub
    #print "v after bound", v
    return v

def soft_thres(v,s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    u=v.copy()
    u[u<0]=0.0
    return np.clip(u-s,0.0,None)

def euclidean_proj_simplex_k(v, k=None):
    '''Tasos's method'''
    n, = v.shape # will raise ValueError if v is not 1-D
    if (k==None):
        k=n
    vid=np.argsort(v)[::-1][:k];
    w=np.zeros(v.shape)
    tau=(1.0/k)*(np.sum(v[vid])-1)
    w[vid]=(v[vid]-tau).clip(min=0)
    return w


def proj_support_bound(v,s_vec=None,lb = 0.0, ub = 1.0):
    #print "v:", v, len(v)
    #print "svec:", s_vec, len(s_vec)
    #print "assert:", len(v)==len(s_vec)
    # v: Input vector
    # s_vec: Binary vector imposing support constraints
    assert lb < ub, "Bound constraints improper!"
    if s_vec==None:
        s_vec = np.ones(v.shape)
    
    assert len(v)==len(s_vec), "input vector and support vector dimensions do not match!"
    assert np.all([(x==0 or x==1) for x in s_vec]), "Function expects s_vec to be a binary vector"
    #print "v before", v    
    v = np.multiply(v,s_vec) # element-wise multiplication
    #print "v after", v
    #print "svec",s_vec
    v[v < lb] = lb
    v[v > ub] = ub
    #print "v after bound", v
    return v

def soft_thres(v,s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    u=v.copy()
    u[u<0]=0.0
    return np.clip(u-s,0.0,None)

def euclidean_proj_simplex_k(v, k=None):
    '''Tasos's method'''
    n, = v.shape # will raise ValueError if v is not 1-D
    if (k==None):
        k=n
    vid=np.argsort(v)[::-1][:k];
    w=np.zeros(v.shape)
    tau=(1.0/k)*(np.sum(v[vid])-1)
    w[vid]=(v[vid]-tau).clip(min=0)
    return w

    
