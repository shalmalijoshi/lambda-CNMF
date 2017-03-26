"""
@author: suriya
"""
import numpy as np
import scipy.linalg as la
from numba import jit
# for code readability
def computeUpdate(X,step,gradX, projX, projX_kwargs={}):    
    if not(isinstance(X,list)):
        Xnew = projX(X-step*gradX,**projX_kwargs)
        Gt = X-Xnew
        Gt2 = la.norm(Gt)**2
        m = np.sum(gradX*Gt)
        return Xnew,m,Gt2
    else:
        Xnew=[];Gt=[];Gt2=0;m=0
        for i in range(len(X)):
            Xnew +=projX[i](X[i]-step*gradX[i],**projX_kwargs)
            Gt += (X[i]-Xnew[i])
            Gt2 = Gt2 + np.sum(Gt[i]*Gt[i])
            m = m + np.sum(gradX[i]*Gt[i])
        return Xnew,m,Gt2

def computeExpUpdate(X, step, gradX):
    Xnew=np.zeros(X.shape)
    if len(X.shape)>1:
        for  i in range(X.shape[0]):            
            Xnew[i,X[i,:]>0]=X[i,X[i,:]>0]*ri
            Xnew[i,:]=Xnew[i,:]/np.sum(Xnew[i,:])
    else:
        ri=np.exp(-step*gradX[X>0])
        Xnew[X>0]=X[X>0]*ri
        Xnew=Xnew/np.sum(Xnew)
        
    Gt = X-Xnew
    Gt2 = la.norm(Gt)**2
    m = np.sum(gradX*Gt)
    return Xnew,m,Gt2

def ProjectedGradientStep(X, D, projX, step=1.0, linesearch=1, tau = 0.3,  verbose=1, D_kwargs={}, projX_kwargs={}, gradIt=10):
    '''
    Inputs: 
    X: variable to be updated in a projected gradient step
    D(X,D_kwargs): divergence function, which returns function values and gradient 
    ProjX(X,projX_kwargs): Computes projection of X
    step:initial step size for PGD
    tau: line search parameter
    linesearch: perform linesearch?
    '''
    #print la.norm(X-projX(X,**projX_kwargs))

    ftol=1e-2
    Xcurr=X.copy()
    # Algorithm starts here
    for i in range(gradIt):
        chtol=1e-8*la.norm(Xcurr)**2
        # Compute Gradients
        f,gradX = D(Xcurr, g=1, **D_kwargs)
        Xnew,m,Gt2 = computeUpdate(Xcurr, step, gradX, projX, projX_kwargs)
        fnew = D(Xnew, g=0, **D_kwargs)
        #print "init fnew", fnew
        if not(linesearch):
            change = np.sqrt(Gt2)
            continue

        # LINE SEARCH 
        # Increase step size
        case=-1
        for k in range(100):
            #if (f-fnew <= max(m-(0.5/step)*Gt2,0.0) or ftmp2==fnew):
            if m-(0.5/step)*Gt2<0:
                print 'm:',m,'step:',step,'Gt2:',Gt2,'0.5/step*Gt2',(0.5/step)*Gt2
            #print "inc", 'k:',k, 'step:',step,'f',f,'fnew',fnew,'f-fnew',f-fnew,'m-0.5/step*Gt2',m-(0.5/step)*Gt2
            if (f-fnew <= max(m-(0.5/step)*Gt2,0.0)):
                case=1
                break
            # Saving latest valid step
            Xtemp=Xnew
            ftemp=fnew
            mtemp=m
            Gt2temp=Gt2
            
            #ftmp2=ftemp
            # Increment
            step=step/tau
            Xnew,m,Gt2 = computeUpdate(Xcurr,step,gradX,projX,projX_kwargs)
            fnew = D(Xnew, g=0, **D_kwargs)
            if (np.sqrt(Gt2)<1e-20):
                case=2
                break
        if k:
            step=step*tau
            Xnew=Xtemp
            fnew=ftemp
            m=mtemp
            Gt2=Gt2temp
        else:
            # Decrease step size
            ftemp = np.inf; Xtemp=Xcurr.copy(); mtemp=m; Gt2temp=Gt2
            for k in range(k,50):
                if m-(0.5/step)*Gt2<0:
                    print 'm:',m,'step:',step,'Gt2:',Gt2,'0.5/step&Gt2:',(0.5/step)*Gt2
                #print "dec",k, f-fnew,m-(0.5/step)*Gt2
                #print "dec", 'k:',k,'step:',step ,'f',f,'fnew',fnew,'f-fnew',f-fnew,'m-0.5/step*Gt2',m-(0.5/step)*Gt2
                if (f-fnew >= max(m-(0.5/step)*Gt2,0.0)):
                    case=3
                    break
                if (Gt2<1e-50):
                    Xnew=Xcurr
                    fnew=f
                    case=4
                    break            
                step = tau*step
                Xnew,m,Gt2 = computeUpdate(Xcurr,step,gradX,projX,projX_kwargs)
                fnew = D(Xnew, g=0, **D_kwargs)
                if (ftemp<fnew and f-ftemp>max(1e-6*m,0)):
                    step = step/tau
                    Xnew = Xtemp
                    fnew = ftemp
                    m = mtemp
                    Gt2 = Gt2temp
                    case=5
                    break
                else:
                    Xtemp = Xnew
                    ftemp = fnew
                    mtemp = m
                    Gt2temp = Gt2

            if (k>=50):
                if verbose>0: print("k>=50,step=%0.2g" %step)
                Xnew=Xcurr
                fnew=f
                Gt2=0

        change = Gt2

        if (verbose>0):
            print('PGDUpdate: change=%0.4g, k=%d, exit_case=%d, step=%0.4g, fnew=%0.4g, f=%0.4g,f-fnew=%0.4g>= %0.4g' \
            %(change, k, case, step, fnew, f, f-fnew, m-(0.5/step)*Gt2))#sigma*m
            
        if change<chtol:
            if verbose:
                print ("Exiting PGD update in %d iterations due to small update change %f" %(i+1,change))
            break
            
        if f-fnew<ftol:
            if verbose:
                print ("Exiting PGD update in %d iterations due to small f change %f" %(i+1,f-fnew))
            break
        
        Xcurr=Xnew.copy()
        
        if verbose and (i==(gradIt-1)):
            print ("Exiting PGD update in %d iterations" %(i+1))
    return Xnew, fnew, step

def ExpoenentiatedGradientStep(X, D, step=1.0, linesearch=1, tau = 0.3,  verbose=1, D_kwargs={}, gradIt=10):
    '''
    Inputs: 
    X: variable to be updated in a exponentiated gradient step 
    if X is a matrix exponentiated gradient step is applied on each row of X

    D(X,D_kwargs): divergence function, which returns function values and gradient 
    step:initial step size for PGD
    tau: line search parameter
    linesearch: perform linesearch?
    '''
    chtol=1e-8*la.norm(X)**2
    ftol=1e-2
    Xcurr=X.copy()
    # Algorithm starts here
    for i in range(gradIt):
        # Compute Gradients
        f,gradX = D(Xcurr, g=1, **D_kwargs)
        Xnew,m,Gt2 = computeExpUpdate(Xcurr, step, gradX)
        fnew = D(Xnew, g=0, **D_kwargs)

        if not(linesearch):
            change = np.sqrt(Gt2)
            continue

        # LINE SEARCH 
        # Increase step size
        case=-1
        for k in range(100):
            if (f-fnew <= max(m-(0.5/step)*Gt2,0.0)):
                case=1
                break
            # Saving latest valid step
            Xtemp=Xnew
            ftemp=fnew
            mtemp=m
            Gt2temp=Gt2

            step=step/tau
            Xnew,m,Gt2 = computeExpUpdate(Xcurr,step,gradX,projX,projX_kwargs)
            fnew = D(Xnew, g=0, **D_kwargs)
            if (np.sqrt(Gt2)<1e-20):
                case=2
                break

        if k:
            step=step*tau
            Xnew=Xtemp
            fnew=ftemp
            m=mtemp
            Gt2=Gt2temp
        else:
            # Decrease step size
            ftemp = np.inf; Xtemp=X; mtemp=m; Gt2temp=Gt2
            for k in range(k,100):
                if (f-fnew >= max(m-(0.5/step)*Gt2,0.0)):
                    case=3
                    break
                if (Gt2<1e-50):
                    Xnew=X
                    fnew=f
                    case=4
                    break            
                step = tau*step
                Xnew,m,Gt2 = computeExpUpdate(Xcurr,step,gradX,projX,projX_kwargs)
                fnew = D(Xnew, g=0, **D_kwargs)
                if (ftemp<fnew and f-ftemp>max(1e-6*m,0)):
                    step = step/tau
                    Xnew = Xtemp
                    fnew = ftemp
                    m = mtemp
                    Gt2 = Gt2temp
                    case=5
                    break
                else:
                    Xtemp = Xnew
                    ftemp = fnew
                    mtemp = m
                    Gt2temp = Gt2

        if (k>=50):
            if verbose>0: print("k>=50,step=%0.2g" %step)
            Xnew=X
            fnew=f
            Gt2=0

        change = Gt2

        if (verbose>0):
            print('EGDUpdate: change=%0.4g, k=%d, exit_case=%d, step=%0.4g, fnew=%0.4g, f=%0.4g,f-fnew=%0.4g>= %0.4g' \
            %(change, k, case, step, fnew, f, f-fnew, m-(0.5/step)*Gt2))#sigma*m
            
        if change<chtol:
            if verbose:
                print ("Exiting EGD update in %d iterations due to small update change %f" %(i+1,change))
            break
            
        if f-fnew<ftol:
            if verbose:
                print ("Exiting EGD update in %d iterations due to small f change %f" %(i+1,f-fnew))
            break
        
        Xcurr=Xnew.copy()
        
        if verbose and (i==(gradIt-1)):
            print ("Exiting EGD update in %d iterations" %(i+1))
    return Xnew, fnew, step
