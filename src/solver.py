'''
Created on Dec 29, 2012

@author: dstrauss
Copyright 2013 David Strauss

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


method definition for doing CG solves with a pipelined matrix-vector multiplication

'''

import numpy as np


def cg(A,b,maxiter=30,tol=1e-6,pll=False):
    ''' run CG iterations in order to solve the equation Ax=b,
    A ==> a function that implements "matrix vector" multiplication, must be Positive Semidefinite
    b ==> right hand side in Ax=b
    maxiter ==> maximum number of CG iterations
    tol ==> exit tolerance, if ||Ax-b|| < tol the program exits
    pll ==> boolean flag for doing plotting or not
    '''
    
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
    
    