'''
Created on Dec 26, 2012

@author: dstrauss

class for implementing the l1 update via pipelined admm

'''

import numpy as np
import scipy.sparse.linalg as lin
import solver

class lasso(object):
    ''' class to implement l1 minimization update '''
    def __init__(self,m,n,rho,lmb):
        ''' set some parameters '''
        self.m = m
        self.n = n
        self.rho = rho
        self.lmb = lmb
        
        self.zp = np.zeros(self.n,dtype='complex128') # primal variable
        self.zd = np.zeros(self.n,dtype='complex128') # aux variable
#        self.zt = np.zeros(self.n,dtype='complex128') # dual variable
        
        
    def solveL1(self,y,A):
        ''' solve min ||y-Ax|| + lmb||x||_1 with a warm start for x=zt 
        input: x,y,A
        where A is a designerConv/convFourier object with mtx, mtxT routines 
        '''
        zt = np.zeros(self.n,dtype='complex128')
#        zd = np.zeros(self.n,dtype='complex128')
        
        Atb = A.mtxT(y);
        M = invOp(A,self.rho,self.m)
        self.rrz = list()
        self.gap = list()
        
        for itz in range(20):
            b = Atb + self.rho*(self.zd-zt);
            
            ss,info = solver.cg(M,A.mtx(b),tol=1e-6,maxiter=20)
            
            print 'l1 iter: ' + repr(itz) + ' converge info: ' + repr(info)
            self.zp = b/self.rho - (1.0/(self.rho**2))*(A.mtxT(ss))
            
            
            self.zd = svt(self.zp+ zt,self.lmb/self.rho)
    
            zt = zt + self.zp-self.zd
            
            self.rrz.append(np.linalg.norm(A.mtx(self.zp) - y))
            self.gap.append(np.linalg.norm(self.zp-self.zd ))
            
        return self.zd
        
        

def invOp(fnc,rho,n):
    '''create an object that does a simple algebraic multiplication'''
    return lambda x: x + (1.0/rho)*fnc.mtx(fnc.mtxT(x))
#        return lin.LinearOperator((n,n), lambda x: x + (1.0/rho)*fnc.mtx(fnc.mtxT(x)), \
#                                  dtype='complex128')
    
    
def svt(z,lmb):
    ''' soft thresholding '''
    return np.maximum(1.0-lmb/np.abs(z),0.0)*z;

def testSvt():
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5,5,500)
    y = svt(x,2)
    plt.figure(1)
    plt.plot(x,y)
    plt.show()
    
    return y
    