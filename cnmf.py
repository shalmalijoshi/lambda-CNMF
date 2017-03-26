import numpy as np
import optimization_numbafuncs as opt
from ad_nmf_helper import factorDivergenceFunction,parfactorDivergenceFunction,projWsupport,projBound
import scipy.linalg as la
import scipy.sparse as sp
import time
from numba import jit
from utils import D_spPoisson
from multiprocessing import Pool, Array, Value
class KeyboardInterruptError(Exception): pass

tauA=0.3
tauW=0.3 # line search parameter

w_bias=0
#X=WA.T alternating updates between W and A
def constrained_nmf_par(X, projW, projA, Dcol, W, A, bias=0, numIt=50, gradIt=10, debug=0, verbose=0,func_kwargs={},supportW=None):
    try:
        print("spawning worker threads")
        p_pool = Pool(14)
        print("done spawning worker threads")
        parfunc = lambda *args,**kwargs: parfactorDivergenceFunction(*args,p_pool=p_pool,**kwargs)
        W,A,fval,stats=constrained_nmf_iterations(X, projW, projA, Dcol, parfunc, W, A, bias, numIt, gradIt, debug, verbose,func_kwargs)
        p_pool.close()
        return W,A,fval,stats

    except KeyboardInterrupt:
        p_pool.terminate()
    except Exception as e:
        p_pool.terminate()
        import sys,traceback
        print traceback.print_tb(sys.exc_info()[2])
        raise e
    finally:   
        p_pool.join()

#X=WA.T alternating updates between W and A
def constrained_nmf_serial(X, projW, projA, Dcol, W, A, bias=0, numIt=50, gradIt=10, debug=0, verbose=0,func_kwargs={}):
    return constrained_nmf_iterations(X, projW, projA, Dcol, factorDivergenceFunction, W, A, bias, numIt, gradIt, debug, verbose,func_kwargs)

#X=WA.T alternating updates between W and A
def constrained_nmf_iterations(X, projW, projA, Dcol, factorDivergenceFunction, W, A, bias=0, numIt=50, gradIt=10, debug=0, verbose=0,func_kwargs={}):
    ftol=1e-3*np.prod(X.shape)

    stats=dict()    
    if sp.issparse(X):
        Xt=X.T.tocsc()
        if (not(sp.isspmatrix_csc(X))):
            X=X.tocsc()
    else:
        Xt=X.T


    # PGD updates    
    stepA=1
    stepW=1
    if (debug): stats={'fiter':[],'Median_nnz':[], 'nnz':[]};         
    ftmp=np.inf    
    if not(w_bias): W[:,-1]=0.0
    Dcol_kwargs=func_kwargs.get('Dcol',{})  

    for i in range(numIt):

        if stepA<0.0001: stepA = 1
        if stepW<0.0001: stepW = 1

        chtol=1e-6*(la.norm(A)**2+la.norm(W)**2)
        Wtemp=W.copy()
        Atemp=A.copy()
        change=0.0

        print("update W")
        if bias:
            Ab = np.hstack((A[:,:-1],w_bias*np.ones((A.shape[0],1))))
            b = A[:,-1]
        else:
            if verbose:
                print("constrained_nmf: no bias")
            Ab=A
            b=0.0
        
        Ab_sum = np.sum(Ab,0)
        b_sum = np.sum(b)
        Dcol_kwargs.update({'A_sum':Ab_sum,'b_sum':b_sum})
        projW_kwargs=func_kwargs.get('projW',{});
        projW_kwargs.update({'w_bias':w_bias})
        DW_kwargs={'U':Ab,'X':Xt,'b':b,'Dcol':Dcol,'Dcol_kwargs':Dcol_kwargs}#,'support':func_kwargs.get('projW',{}).get('support')}
        
        W,fval,stepW = opt.ProjectedGradientStep(X=W, D=factorDivergenceFunction, projX=projW, step=stepW, tau = tauW, linesearch=1, verbose=verbose, D_kwargs=DW_kwargs, projX_kwargs=projW_kwargs,gradIt=gradIt)
        if debug: stats['fiter'].append(fval);
        
        ######################################
        print("update A")
        if bias: 
            Wb = np.hstack((W[:,:-1],np.ones((W.shape[0],1))))
            b = W[:,-1]
        else: 
            Wb=W
            b=0

        Wb_sum = np.sum(Wb,0)
        b_sum = np.sum(b)
        Dcol_kwargs.update({'A_sum':Wb_sum,'b_sum':b_sum}) 
        DA_kwargs={'U':Wb,'X':X,'b':b,'Dcol':Dcol,'Dcol_kwargs':Dcol_kwargs}#,'support':func_kwargs.get('projA',{}).get('support')}
        
        A,fval,stepA = opt.ProjectedGradientStep(X=A, D=factorDivergenceFunction, projX=projA, step=stepA, tau = tauA, linesearch=1, verbose=verbose,D_kwargs=DA_kwargs, projX_kwargs=func_kwargs.get('projA',{}),gradIt=gradIt)
        
        if debug:
            nnz=[len(np.where(A[:,j]>=1e-15)[0]) for j in range(A.shape[1]-bias)]
            stats['Median_nnz'].append(np.median(nnz))
            stats['nnz'].append(np.array(nnz))
            stats['fiter'].append(fval);
            
        #########################################           
        
        print("\t Iter:%d. fval=%0.4g. Phenotype Sparsity: median(nnz)=%d/%d, (min_nnz,max_nnz)=(%d,%d)"\
              %(i, fval,np.median(nnz), A.shape[0], np.min(nnz), np.max(nnz)))
        
        change = la.norm(W-Wtemp)**2+la.norm(A-Atemp)**2
        if ((i>=10) and ((change < chtol) or  (ftmp-fval<ftol))):
            print("Exited in %d iterations due to insufficient change. Loss_value=%f" % (i,fval))
            break
                    
    if (i==numIt-1):
        print("Exited due to max_iter with ch=%f, loss_value=%f" % (change, fval))
        
    if debug: stats['niter'] = i

    return W,A,fval,stats

#X=AW.T
def computeW(X,A,Dcol=D_spPoisson,projW=projBound,W=None,Dcol_kwargs={},projW_kwargs={},bias=1):
    n=X.shape[1]
    rk=A.shape[1]-bias
    
    if W is None:
        W=projW(np.random.rand(n,rk+bias),**projW_kwargs)
        if bias: W[:,-1]=w_bias*W[:,-1]
    stepW=1    
    
    
    if bias: 
        Ab = np.hstack((A[:,:-1],w_bias*np.ones((A.shape[0],1))))
        b = A[:,-1]
    else:
        Ab=A
        b=0.0
    Ab_sum = np.sum(Ab,0)
    b_sum = np.sum(b)
    Dcol_kwargs.update({'A_sum':Ab_sum,'b_sum':b_sum})
    projW_kwargs.update({'w_bias':w_bias})
    DW_kwargs={'U':Ab,'X':X,'b':b,'Dcol':Dcol,'Dcol_kwargs':Dcol_kwargs}
    Wtmp = W.copy()
    W,fval,_= opt.ProjectedGradientStep(X=W, D=factorDivergenceFunction, projX=projW, step=stepW, tau = tauW, linesearch=1, verbose=1, D_kwargs=DW_kwargs, projX_kwargs=projW_kwargs,gradIt=50)
    
    print "Finished Projection: fval=%f" %fval

    
    return W,Wtmp
