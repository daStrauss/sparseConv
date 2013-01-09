'''
Created on Dec 26, 2012

@author: dstrauss

routines for updating local weights

'''

import scipy.sparse.linalg as lin
import numpy as np
import scipy.signal as sig
import solver

class weightsUpdate(object):
    ''' class that implements methods for updating weights '''
    def __init__(self,m,p,q,xi):
        self.p = p
        self.q = q
        self.m = m
        self.n = self.q*self.p
        self.xi = xi
        self.wp = np.zeros((q,p))
        self.wd = np.zeros((q,p))
 
        
    def updatePlain(self,y,wt,z):
        ''' update for weights '''
        self.wd = self.wd + (self.wp-wt);
        self.zm = z.reshape(self.m,self.p,order='F')

        M = self.makeCGM()
        b = self.mtxT(y) + self.xi*(wt.flatten(order='F')-self.wd.flatten(order='F'))

        ss,info = solver.cg(M,b,tol=1e-6,maxiter=30,pll=False)
        print 'CG info: ' + repr(info)
        
        w = ss.real.reshape(self.q,self.p,order='F') 

        self.wp = w
        return (w+self.wd)
    
    def mtx(self,x):
        ''' multiply method for z*w '''
        xl = x.reshape(self.q,self.p,order='F')
        
        y = np.zeros(self.m,x.dtype)
            
        slc = slice(self.q/2,self.q/2+self.m)
        for ix in range(self.p):
            tmp = sig.fftconvolve(xl[:,ix], self.zm[:,ix].flatten())
            y += tmp[slc]
        
        return y
    
    def mtxT(self,y):
        ''' adjoint multiplication operator '''
        assert(y.size==self.m)
        
        x = np.zeros(self.n,y.dtype)
        # print 'y type ' + repr(y.dtype)
        slc = slice(self.m - self.q/2-1,self.m + self.q/2-1)
            
        for ix in range(self.p):
            slz = slice((ix*self.q),(ix*self.q)+self.q)
            tmp = sig.fftconvolve(y,np.flipud(self.zm[:,ix].flatten()))
            x[slz] = tmp[slc]
               
        return x
                
        
    def updateFourier(self,y,wt,z):
        '''update method for when there are fourier modes too '''
        self.wd = self.wd + (self.wp-wt);
        self.zm = z[:(self.m*self.p)].reshape(self.m,self.p,order='F')
        zf = z[(self.m*self.p):]
        
        M = self.makeCGM()
        print y.shape
        print wt.shape
        print z.shape
        print zf.shape
        print self.m
        print self.p
        b = self.mtxT(y - (1.0/np.sqrt(self.m))*np.fft.fft(zf)) + self.xi*(wt.flatten(order='F')-self.wd.flatten(order='F'))

        ss,info = solver.cg(M,b,tol=1e-6,maxiter=30,pll=False)
        print 'CG info: ' + repr(info)
        
        w = ss.real.reshape(self.q,self.p,order='F') 

        self.wp = w
        return (w+self.wd)

    

    def makeCGM(self):
        ''' make linear operator with AtA '''
        return lambda x: self.xi*x + self.mtxT(self.mtx(x))
#    
#        return lin.LinearOperator((self.p*self.q,self.p*self.q), lambda x: self.xi*x + self.mtxT(self.mtx(x)), \
#                                    dtype='complex128')
#        
        
        
        
def test():
    import scipy.io as spio
    import matplotlib.pyplot as plt
    
    D = spio.loadmat('fakeUP.mat')
    m = D['m']
    p = D['p']
    q = D['q']
    bt = D['b'].flatten()
    xi = 0.2
    
    W = weightsUpdate(m,p,q,xi)
    wt = np.zeros(D['wTrue'].shape)
    y = D['sig'].flatten()
    z = D['z'].flatten()
    
    print wt.shape
    print y.shape
    
    ww = W.updatePlain(y, wt, z)
    ww = ww.flatten()
    b = W.mtxT(y)
    
    plt.figure(83)
    plt.plot(range(b.size),b,range(bt.size),bt)
    
    plt.figure(10)
    plt.plot(range(ww.size),ww, range(ww.size),D['wTrue'].flatten())
    plt.show()
    
    
if __name__=='__main__':
    test()
    
    