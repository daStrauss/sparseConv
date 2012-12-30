'''
Created on Dec 29, 2012

@author: dstrauss
'''

import numpy as np


def cg(A,b,maxiter=30,tol=1e-6,pll=False):
    x = np.zeros(b.size,dtype=b.dtype)
    
    r=b-A(x)
    p=r
    rsold=np.dot(r.T,r)
    rsn = list()
    
    ix = 0
    while ix < maxiter:
        ix += 1
        Ap = A(p);
        alpha=rsold/np.dot(p.T,Ap);
        x=x+alpha*p;
        r=r-alpha*Ap
        rsnew = np.dot(r.T,r) 
        
        rsn.append(np.sqrt(rsnew))
        if np.sqrt(rsnew)<tol:
            break
          
        p = r+ (rsnew/rsold)*p;
        rsold=rsnew;

    if pll:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(rsn)
        plt.title('rsn')
        plt.show()
            
    return x,ix


def test():
    import scipy.sparse 
    
    opr = -np.ones((3,20))
    opr[1,:] = 2
        
    M = scipy.sparse.spdiags(opr, [-1,0,1], 20,20);
    
    b = np.zeros(20)
    b[9] = 1
    
    cg(lambda x: M*x,b)
    
    