'''
Created on Dec 26, 2012

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

class for implementing the l1 update via pipelined admm

'''

import numpy as np
import solver
import copy

class lasso(object):
    ''' class to implement l1 minimization update '''
    def __init__(self,m,n,rho,lmb,ch):
        ''' set some parameters '''
        self.m = m 
        self.n = n
        self.ch = ch # number of channels of simultaneous data
        self.rho = rho # internal algorithm parameter 
        self.lmb = lmb # fixed regularization parameter
        self.alp = 1.5 # over-relaxation parameter
        
        self.zp = np.zeros(self.n,dtype='complex128') # primal variable
        self.zd = np.zeros(self.n,dtype='complex128') # aux variable
#        self.zt = np.zeros(self.n,dtype='complex128') # dual variable
        
        
    def solveL1(self,y,A):
        ''' solve min ||y-Ax|| + lmb||x||_1 with a warm start for x=zt 
        input: x,y,A
        where A is a designerConv/convFourier object with mtx, mtxT routines 
        '''
        # make sure that the input data and the transformation is the correct size
        assert(len(y) == self.ch)
        assert(len(A) == self.ch)
        
        zt = [np.zeros(self.n,dtype='complex128') for ix in xrange(self.ch)]
        zd = [np.zeros(self.n,dtype='complex128') for ix in xrange(self.ch)]

        # debug by println statements -- gives a flavor of how the listing brackets work
#        print [yl.shape for yl in y]
        Atb = [Al.mtxT(yl) for Al,yl in zip(A,y)]
#        print [Atbl.shape for Atbl in Atb]
#        print [ztl.shape for ztl in zt]
        
        M = [invOp(Al,self.rho,self.m) for Al in A]
        
        self.rrz = list()
        self.gap = list()
        
        for itz in range(20):
            
            b = [Atbl + self.rho*(zdl-ztl) for Atbl,zdl,ztl in zip(Atb,zd,zt)]
            
            sout = [solver.cg(Ml,Al.mtx(bl),tol=1e-6,maxiter=20) for Ml,Al,bl in zip(M,A,b)]
            
            stg = 'l1 iter: ' + repr(itz)
            for ss in sout:
                stg += ' cvg ' + repr(ss[1])
            
            zold = copy.deepcopy(zd)
            
            # project into linear constraint
            uux = [bl/self.rho-(1.0/(self.rho**2))*(Al.mtxT(ss[0])) for bl,Al,ss in zip(b,A,sout)]
            
            # over relax
            zp = [self.alp*uuxl + (1.0 - self.alp)*zoldl for uuxl,zoldl in zip(uux,zold)]
            
            # soft threshold
            zths = [a+b for a,b in zip(zp,zt)]
            zd = svtspecial(zths,self.lmb/self.rho)
            
            # update dual variables
            zt = [a + b-c for a,b,c in zip(zt,zp,zd)]
            
            self.rrz.append(sum([np.linalg.norm(Al.mtx(zpl) - yl) for Al,zpl,yl in zip(A,zp,y)]))
            self.gap.append(sum([np.linalg.norm(zpl-zdl) for zpl,zdl in zip(zp,zd)] ))
            
        return zd
        
        

def invOp(fnc,rho,n):
    '''create an object that does a simple algebraic multiplication'''
    return lambda x: x + (1.0/rho)*fnc.mtx(fnc.mtxT(x))
#        return lin.LinearOperator((n,n), lambda x: x + (1.0/rho)*fnc.mtx(fnc.mtxT(x)), \
#                                  dtype='complex128')
    
    
def svt(z,lmb):
    ''' soft thresholding '''
    return np.maximum(1.0-lmb/np.abs(z),0.0)*z;

def svtspecial(z,lmb):
    ''' soft thresholding '''
    return np.maximum(1.0-lmb/np.sqrt(sum(np.abs(z)**2)),0.0)*z;

def testSvt():
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5,5,500)
    y = svt(x,2)
    plt.figure(1)
    plt.plot(x,y)
    plt.show()
    
    return y


def test():
    ''' test against matlab routine. double check! because there ain't no CVX for this'''
    
    import scipy.io as spio
    import matplotlib.pyplot as plt
    import designerConv
    import convFourier
    
    # load data from a fakel1 data set created by 'testL1.m'
    D = spio.loadmat('fakeL1.mat')
    m = D['m'].astype('int64').flatten()
    p = D['p'].astype('int64').flatten()
    q = D['q'].astype('int64').flatten()
    wTrue = D['wTrue']
    y = D['sig'].flatten()
    
    print 'size of m,p,q ' + repr(m) + ' ' + repr(p) + ' ' + repr(q)
    print 'and thes is mp ' + repr(m*p)
    rho = 1
    lmb = 0.2
    W = lasso(m,(p+1)*m,rho,lmb)
    
#    A = designerConv.convOperator(m,p,q)
    A = convFourier.convFFT(m,p,q)
    A.changeWeights(wTrue)
    
    
    print wTrue.shape
    print y.shape
    
    z = W.solveL1(y, A)
    
    zTrue = D['zTrue'].flatten()
    zM = D['zd'].flatten()
    
    print np.linalg.norm(zM-z)
    
    plt.figure(83)
    plt.plot(range(z.size), np.abs(z), range(zTrue.size), np.abs(zTrue),
             range(zM.size), np.abs(zM))
    
    plt.show()
    

if __name__=='__main__':
    test()
    