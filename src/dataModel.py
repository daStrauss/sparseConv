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
    def __init__(self,m,p,q,ch,fourier=True):
        self.m = m
        self.p = p
        self.q = q
        self.ch = ch
        self.fourier = fourier
        if self.fourier:
            self.n = self.m*(self.p+1)
        else:
            self.n = self.m*self.p
                
    
    def changeWeights(self,wn):
        assert (len(wn) == self.ch)
        self.w = wn
        
        for w in self.w:
            assert(w.shape[0] == self.q)
            if self.p > 1:
                assert(w.shape[1] == self.p)
    
    def applyW(self,zin):
        ''' apply all of the convolutional weights to the same set of z '''
        y = list()
        for w,z in zip(self.w,zin):
            xl = z[:(self.m*self.p)].reshape(self.m,self.p,order='F')
            slc = slice(self.q/2,self.q/2+self.m)
            mpx = mapFFTConv(slc)
            y.append(sum(map(mpx, zip(xl.T,w.T))))

        if self.fourier:
            for yl,z in zip(y,zin):
                slz = slice(self.m*self.p,self.m*self.p + self.m) 
                yl += (1.0/np.sqrt(self.m))*np.fft.fft(z[slz])
        return y
    
    
    def applyWT(self,y):
        ''' adjoint multiplication operator '''
        assert(len(y)==self.ch)
        z = list()
        
        for w,y in zip(y):
            x = np.zeros(self.n,y.dtype)
            for ix in range(self.p):
                slz = slice((ix*self.m),(ix*self.m)+self.m)
                x[slz] = sig.fftconvolve(y,np.flipud(self.w[:,ix].flatten()),'same')
            z.append(x)
            
        return z
    
    def mtx(self,zin):
        
    

def test():
    import designerConv
    import time
    import matplotlib.pyplot as plt
    p = 5
    q = 10
    m = 1000
    
    ''' create the operator, initialize '''
    A = designerConv.convOperator(m,p,q)
    w = [np.random.randn(q,p)/np.sqrt(q), np.random.randn(q,p)/np.sqrt(q)]
    A.changeWeights(w[0])
    
    C = designerConv.convOperator(m,p,q)
    C.changeWeights(w[1])
    '''create random x'''
    x = [np.random.randn(m*p), np.random.randn(m*p)]
    
    
    
    B = dtm(m,p,q,2,fourier=False)
    B.changeWeights(w)
    
        
    ''' apply functional operator and its transpose '''
    tm = time.time()
    yf = [A.mtx(x[0]), C.mtx(x[1])]
    print 'single itme ' + repr(time.time()-tm)
    
    tm = time.time()
    yp = B.applyW(x)
    print 'new time ' + repr(time.time()-tm)
    
    print 'yf s ' + repr(yf[0].shape)
    print 'yp s ' + repr(yp[0].shape)
    
    for a,b in zip(yf,yp):
        print 'diff ' + repr(np.linalg.norm(a-b))
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(range(yf[0].size), yf[0].real, np.arange(yp[0].size), yp[0].real)
    
    plt.subplot(2,1,2)
    plt.plot(range(yf[1].size), yf[1].real, np.arange(yp[1].size), yp[1].real)

    
    plt.show()
    
    return yp
    
if __name__ == '__main__':
    test()
                
                