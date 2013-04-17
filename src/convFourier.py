'''
Created on Dec 25, 2012

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


inelegantly extends designer conv so that it also calculates the FFT

'''

from designerConv import convOperator as cvOp
import numpy as np

class convFFT(object):
    ''' implements the convolution, but augments and extends to include the 
    fourier family '''
    
    def __init__(self,m,p,q,fct=1.0):
        ''' initialization for convolution operator
        creates an m x n implicit matrix
        m is the length of the 'data to reconstruct' 
        q is the length of the filters 
        p is the number of filters '''
        
        self.m = m
        self.p = p
        self.q = q
        self.W = cvOp(m,p,q)
        self.fct = fct
        
        self.n = self.m*self.p+self.m # need to add one in order to get true dimensions
        
        
    def changeWeights(self,w):
        ''' reset internal weights '''
        self.W.changeWeights(w)
        
    def mtx(self,x):
        ''' multiplication operator '''
        assert(x.size == self.n)
        
        y = self.W.mtx(x[:(self.m*self.p)]).astype('complex128')
        
        slz = slice(self.m*self.p,self.m*self.p + self.m) 
        y += (self.fct/np.sqrt(self.m))*np.fft.fft(x[slz])
                
        return y
            
    
    def mtxT(self,y):
        ''' adjoint multiplication operator '''
        assert(y.size==self.m)
        
        x = np.zeros(self.n, dtype='complex128')
        slz = slice(0,self.m*self.p)
        x[slz] = self.W.mtxT(y)
        
        slz = slice(self.m*self.p,self.m*(self.p+1))
        x[slz] = np.sqrt(self.m)*np.fft.ifft(y)/self.fct
    
        return x
    
    def matvec(self,x):
        ''' again, implement AtA operator for doing pcg with this matrix '''
        return self.mtxT(self.mtx(x))
    
def test():
    '''test routine to make sure things work '''
    p = 1
    q = 5
    m = 20
    
    ''' create the operator, initialize '''
    A = convFFT(m,p,q)
    w = np.random.randn(q,p)
    A.changeWeights(w)
    
    '''create random x'''
#    x = np.random.randn(2*m).astype('complex128')
#    x[m:2*m] += 1j*np.random.randn(m)
    x = np.zeros(2*m,dtype='complex128')
    x[m:2*m] += np.random.randn(m) + 1j*np.random.randn(m)
    
    print x.dtype
    
    ''' apply functional operator and its transpose '''
    yf = A.mtx(x)
    print 'yf type ' + repr(yf.dtype)
    
    xf = A.mtxT(yf)
    
    return x,yf,xf

def testMulti():
    ''' test routine to make sure multiple filters work '''
    import matplotlib.pyplot as plt
    
    ''' some test dimensions, small in size '''
    p = 20
    q = 10
    m = 50
    
    ''' create the operator, initialize '''
    A = convFFT(m,p,q)
    w = np.random.randn(q,p)/np.sqrt(q)
    A.changeWeights(w)
    
    '''create random x'''
    x = np.random.randn(m*(p+1)) + 1j*np.random.randn(m*(p+1))
    
    
    ''' apply functional operator and its transpose '''
    yf = A.mtx(x)
    xf = A.mtxT(yf)
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(range(m), yf.real)
    
    plt.subplot(2,1,2)
    plt.plot(range(m*(p+1)), x.real, range(m*(p+1)), xf.real)
    
    plt.show()
    
    