'''
Created on Jan 4, 2013

@author: dstrauss

Trying something new:
a modeling framework that can incorporate all of the basic elements.
it will have 
applyW
applyWT
applyZ
applyZT

setW

y
z
w

I want to be able to build one of these that has "multi channel support"

'''
import numpy as np
import scipy.signal as sig

class mapFFTConv(object):
    def __init__(self,slz):
        self.slz = slz
    def __call__(self,a):
        tmp = sig.fftconvolve(a[0],a[1])
        return tmp[self.slz]
    
    
class dtm(object):
    def __init__(self,m,p,q,fourier=True):
        self.m = m
        self.p = p
        self.q = q
        self.fourier = fourier
        if self.fourier:
            self.n = self.m*(self.p+1)
        else:
            self.n = self.m*self.p
                
    
    def changeWeights(self,wn):
        assert(wn.shape[0] == self.q)
        if self.p > 1:
            assert(wn.shape[1] == self.p)
            
        self.w = wn
        
        
    
    def applyW(self,z):
        ''' apply all of the convolutional weights to the same set of z '''
        xl = z[:(self.m*self.p)].reshape(self.m,self.p,order='F')
        slc = slice(self.q/2,self.q/2+self.m)  
        mpx = mapFFTConv(slc)
        y = sum(map(mpx, zip(xl.T,self.w.T)))
            

        if self.fourier:
            ''' add in the fourier components '''
            slz = slice(self.m*self.p,self.m*self.p + self.m) 
            y += (1.0/np.sqrt(self.m))*np.fft.fft(z[slz])
            
        return y
    
    
    def applyWT(self,y):
        ''' adjoint multiplication operator '''
        x = list()
        for wl in self.w.T:
            x.append(sig.fftconvolve(y,np.flipud(wl.flatten()),'same'))    

        if self.fourier:
            ''' compute the fourier coefficients '''
            x.append(np.sqrt(self.m)*np.fft.ifft(y))
            
        x = np.array(x).flatten()
        return x
    
    def solveL1(self,yin):
        ''' solve min ||y-Ax|| + lmb||x||_1 with a warm start for x=zt 
        input: x,y,A
        where A is a designerConv/convFourier object with mtx, mtxT routines 
        '''
        assert(len(yin) == self.ch)
        zt = np.zeros(self.n,dtype='complex128')
        zd = np.zeros(self.n,dtype='complex128')
        
        Atb = A.mtxT(y);
        M = invOp(A,self.rho,self.m)
        self.rrz = list()
        self.gap = list()
        
        for itz in range(20):
            b = Atb + self.rho*(self.zd-zt);
            
            ss,info = solver.cg(M,A.mtx(b),tol=1e-6,maxiter=20)
            
            print 'l1 iter: ' + repr(itz) + ' converge info: ' + repr(info)
            
            zold = self.zd
            
            uux = b/self.rho - (1.0/(self.rho**2))*(A.mtxT(ss))
            
            self.zp = self.alp*uux + (1.0 - self.alp)*zold;
            
            self.zd = svt(self.zp + zt,self.lmb/self.rho)
    
            zt = zt + self.zp-self.zd
            
            self.rrz.append(np.linalg.norm(A.mtx(self.zp) - y))
            self.gap.append(np.linalg.norm(self.zp-self.zd ))
            
        return self.zd
    
        
def test():
    import convFourier
    import time
    import matplotlib.pyplot as plt
    p = 5
    q = 10
    m = 1000
    
    ''' create the operator, initialize '''
    A = convFourier.convFFT(m,p,q)
    w = [np.random.randn(q,p)/np.sqrt(q), np.random.randn(q,p)/np.sqrt(q)]
    A.changeWeights(w[0])
    
    C = convFourier.convFFT(m,p,q)
    C.changeWeights(w[1])
    '''create random x'''
    x = [np.random.randn(m*(p+1)).astype('complex128'), np.random.randn(m*(p+1)).astype('complex128')]
    
    
    B = [dtm(m,p,q,fourier=True) for ix in xrange(2)]
    for Q,wl in zip(B,w):
        Q.changeWeights(wl)

        
    ''' apply functional operator and its transpose '''
    tm = time.time()
    yf = [A.mtx(x[0]), C.mtx(x[1])]
    print 'single itme ' + repr(time.time()-tm)
    
    tm = time.time()
    yp = [Q.applyW(xl) for Q,xl in zip(B,x)] 
    print 'new time ' + repr(time.time()-tm)
    
    print 'yf s ' + repr(yf[0].shape)
    print 'yp s ' + repr(yp[0].shape)
    
    for a,b in zip(yf,yp):
        print 'diff ' + repr(np.linalg.norm(a-b))
        
    tm = time.time()
    g = [A.mtxT(yf[0]), C.mtxT(yf[1])]
    print 'making list time ' + repr(time.time()-tm)
    
    tm = time.time()
    h = B.applyWT(yp)
    print 'new time ' + repr(time.time()-tm)
    
    for a,b in zip(g,h):
        print 'diff ' + repr(np.linalg.norm(a-b))
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(range(yf[0].size), yf[0].real, np.arange(yp[0].size), yp[0].real)
    
    plt.subplot(2,1,2)
    plt.plot(range(yf[1].size), yf[1].real, np.arange(yp[1].size), yp[1].real)

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(range(g[0].size), g[0].real, range(h[0].size), h[0].real)
    
    plt.subplot(2,1,2)
    plt.plot(range(g[1].size), g[1].real, range(h[1].size), h[1].real)
    
    plt.show()
    
    return yp
    
if __name__ == '__main__':
    test()
                
                