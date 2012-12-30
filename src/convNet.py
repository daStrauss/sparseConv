'''
Created on Dec 26, 2012

@author: dstrauss

creating some routines to make a convolutional net. especially one that works in parallel

'''

import scipy.io as spio
import numpy as np
from convFourier import convFFT
from lassoUpdate import lasso
import weightsUpdate
from time import time
from mpi4py import MPI
import sys
from designerConv import convOperator


def main():
    ''' main routine '''
    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    nProc = comm.Get_size()
    
    plain = True
    if len(sys.argv) == 2:
        if sys.argv[1] == 'plain':
            plain = False
        
        
    m = 50000 # size of data
    p = 50 # number of filters
    q = 300 # length of filters
    
    rho = 5.0
    lmb = 0.5e-3
    xi = 0.2
    fac = np.sqrt((m/q)/2.0)
    ''' initialize MPI routine '''
    
    
    ''' load data, given rank '''
    y = getData(m,rank=rk)

    
    ''' initialize weights '''
    # D = spio.loadmat('fakew.mat')
    # wt = D['wini']
    wt = np.random.randn(q,p)/np.sqrt(q) #D['w']
    
    wt = weightAgg(wt,p,q,comm)
    
    if plain:
        A = convFFT(m,p,q,fct=fac)
        optl1 = lasso(m,m*(p+1),rho,lmb)
        
    else:
        print 'Doing plain!'
        A = convOperator(m,p,q)
        optl1 = lasso(m,m*(p),rho,lmb)
        
        
    newWW = weightsUpdate.weightsUpdate(m,p,q,xi,fct=fac)
    newWW.wp = wt
    
    rrz = list()
    gap = list()
    ''' begin loop '''
    for itz in range(1000):
        ws = newWW.wp
        A.changeWeights(newWW.wp)
        tm = time()
        z = optl1.solveL1(y, A)
        rrz = optl1.rrz # .append(optl1.rrz.pop())
        gap = optl1.gap #.append(optl1.gap.pop())
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved l1 itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()
        
        if plain:
            wmx = newWW.updateFourier(y, wt, z)
        else:
            wmx = newWW.updatePlain(y, wt, z)
            
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved w update itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()
        wt = weightAgg(wmx,p,q,comm)
        wp = newWW.wp
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' have new weights itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        outd = {'y':y, 'z':z, 'wt':wt,'wp':wp,'m':m,'p':p,'q':q, 'rho':rho,'lmb':lmb, 'xi':xi, 'rrz':rrz,'gap':gap, 'ws':ws }
        
        if plain:
            spio.savemat('miniOut_' + repr(itz) + '_' + repr(nProc) + '_' + repr(rk), outd) 
        else:
            spio.savemat('plainOut_' + repr(itz) + '_' + repr(nProc) + '_' + repr(rk), outd) 
    
    return y,z,optl1,A,wt
    
    ''' for n iterations '''
    
    ''' calculate local weights '''
    
    ''' all reduce '''
    
    
def weightAgg(U,p,q,comm):
    ''' aggregation for weights '''
    
    wmx = np.zeros((q,p))
    
    wmx = comm.allreduce(U,wmx,op=MPI.SUM)
    
    wmx = wmx/comm.Get_size()
    
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
    
#    import matplotlib.pyplot as plt
    
    D = spio.loadmat('../data/plmr.mat')
    
    # upb = D['fs'].size - m-1
    cix = 1000 + rank*m # np.random.random_integers(0,upb,1)
    slz = slice(cix,cix+m)
    y = D['fs'][slz].astype('complex128').flatten()
    y = y - np.mean(y)
    y = y/np.linalg.norm(y)
    
#    plt.figure(200)
#    plt.plot(y.real)
#    plt.show()
    
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
    plt.plot(range(y.size),y, range(y.size), A.mtx(z).real)
    
    plt.figure(12)
    plt.plot(range(w.shape[0]),w[:,0], range(wt.shape[0]),wt[:,0])
    
    return y,z,optl1,A,wt,w
    
    
if __name__ == '__main__':
    main()
    
    
    