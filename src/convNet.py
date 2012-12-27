'''
Created on Dec 26, 2012

@author: dstrauss

creating some routines to make a convolutional net. especially one that works in parallel

'''

import scipy.io as spio
import numpy as np
from designerConv import convOperator
from lassoUpdate import lasso
import weightsUpdate

def main():
    ''' main routine '''
    m = 100000 # size of data
    p = 128 # number of filters
    q = 64 # length of filters
    
    rho = 5.0
    lmb = 1e-3
    
    ''' initialize MPI routine '''
    
    
    ''' load data, given rank '''
    y = getData(m)

    
    ''' initialize weights '''
    D = spio.loadmat('fakew.mat') 
    wt = np.random.randn(q,p)*0.05 #D['w']
    A = convOperator(m,p,q)
    
    optl1 = lasso(m,m*p,rho,lmb)
    newWW = weightsUpdate.weightsUpdate(m,p,q,0.5)
    
    ''' begin loop '''
    for itz in range(1):
        A.changeWeights(wt)
        z = optl1.solveL1(y, A)
        wmx = newWW.updatePlain(y, wt, z)
        wt = weightAgg(wmx,p,q)
    
    return y,z,optl1,A,wt
    
    ''' for n iterations '''
    
    ''' calculate local weights '''
    
    ''' all reduce '''
    
    
def weightAgg(wmx,p,q):
    ''' aggregation for weights '''
    wt = np.zeros((q,p))
    
    for ix in xrange(p):
        nrm = np.linalg.norm(wmx[:,ix])
        if nrm > 1:
            wt[:,ix] = wmx[:,ix]/nrm
        else:
            wt[:,ix] = wmx[:,ix]
    
    return wt
            
    
def getData(m,rank=0):
    ''' simple function to grab data 
    returns zero mean, normalized data sample
    '''
    
    import matplotlib.pyplot as plt
    
    D = spio.loadmat('../data/plmr.mat')
    
    # upb = D['fs'].size - m-1
    cix = 1000 # np.random.random_integers(0,upb,1)
    slz = slice(cix,cix+m)
    y = D['fs'][slz].astype('complex128').flatten()
    y = y - np.mean(y)
    y = y/np.linalg.norm(y)
    
    plt.figure(200)
    plt.plot(y.real)
    plt.show()
    
    print 'shape of y ' + repr(y.shape)
    return y
    
def testMain():
    import matplotlib.pyplot as plt
    y,z,optl1,A,wt = main()
    
    D = spio.loadmat('fakew.mat') 
    w = D['w']
    
    plt.figure(10)
    plt.plot(z.real)
    
    plt.figure(11)
    plt.plot(range(1000),y, range(1000), A.mtx(z).real)
    
    plt.figure(12)
    plt.plot(range(50),w[:,0], range(50),wt[:,0])
    
    return y,z,optl1,A,wt,w
    
    
    
    