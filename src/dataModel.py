'''
Created on Jan 4, 2013

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
    ''' not sure why I built this class. It essentially
    is only a class that will perform FFT-based convolution, which
    may or may not be wise depending on the size of the objects '''
    def __init__(self,slz):
        self.slz = slz
    def __call__(self,a):
        tmp = sig.fftconvolve(a[0],a[1])
        return tmp[self.slz]
    
    
class dtm(object):
    ''' data model (dtm) class
    initialize with model parameters, m,p,q which dictate 
    m = size of data
    p = # number of filters
    q = # length of filters
    fourier is a switch about whether to include a fourier transform
    dictionary or not.
    '''
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
        ''' basic routine to change the weights in the model '''
        assert(wn.shape[0] == self.q)
        if self.p > 1:
            assert(wn.shape[1] == self.p)
            
        self.w = wn
        
        
    
    def mtx(self,z):
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
    
    
    def mtxT(self,y):
        ''' adjoint multiplication operator '''
        x = list()
        # accomplish "adjoint" multiplication through flipped-weight convolution
        for wl in self.w.T:
            x.append(sig.fftconvolve(y,np.flipud(wl.flatten()),'same'))    

        if self.fourier:
            ''' compute the fourier coefficients '''
            x.append(np.sqrt(self.m)*np.fft.ifft(y))
            
        x = np.array(x).flatten()
        return x
    
        
def test():
    ''' a testing routine to make sure that these objects behave as I expect '''
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
    yp = [Q.mtx(xl) for Q,xl in zip(B,x)] 
    print 'new time ' + repr(time.time()-tm)
    
    print 'yf s ' + repr(yf[0].shape)
    print 'yp s ' + repr(yp[0].shape)
    
    for a,b in zip(yf,yp):
        print 'diff ' + repr(np.linalg.norm(a-b))
        
    tm = time.time()
    g = [A.mtxT(yf[0]), C.mtxT(yf[1])]
    print 'making list time ' + repr(time.time()-tm)
    
    tm = time.time()
    h = [Q.mtxT(yl) for Q,yl in zip(B,yp)]
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
                
                
