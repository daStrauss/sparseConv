'''
Created on Dec 26, 2012

@author: dstrauss

creating some routines to make a convolutional net. especially one that works in parallel
and this one is going to work over multiple channels

'''

import scipy.io as spio
import numpy as np
from dataModel import dtm
from lassoUpdate import lasso
import weightsUpdate
from time import time
from mpi4py import MPI
import sys



def main():
    ''' main routine '''
    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    nProc = comm.Get_size()
    
    plain = True
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'plain':
            plain = False
    
    if len(sys.argv) >= 3:
        dts = sys.argv[2]
    else:
        dts = 'plmr'
        
        
        
    m = 10000 # size of data
    p = 25 # number of filters
    q = 300 # length of filters
    
    if dts == 'plmr':
        rho = 1
        lmb = 0.5
        xi = 0.2
    elif dts == 'mpk':
        rho = 0.1
        lmb = 4e-6
        xi = 0.2
        
    fac = 1.0; # np.sqrt((m/q)/2.0)
    ''' initialize MPI routine '''
    
    ch = 3
    ''' load data, given rank '''
    y = getData(m,dts,ch,rank=rk)
    
    
    
    ''' initialize weights '''
    # D = spio.loadmat('fakew.mat')
    # wt = D['wini']
#    wt = np.random.randn(q,p)/np.sqrt(q) #D['w']
    
    
    wt = [weightInit(p,q,comm) for ix in xrange(ch)]
    
    if plain:
        A = [dtm(m,p,q,fourier=False) for ix in xrange(ch)]
        
    else:
        A = [dtm(m,p,q,fourier=True) for ix in xrange(ch)]
    
    for Q in A:
        print Q.n
            
    optl1 = lasso(m,A[0].n,rho,lmb,ch)
        
        
    newWW = [weightsUpdate.weightsUpdate(m,p,q,xi) for ix in xrange(ch)]
    print newWW[0].m
    for WWl,wl in zip(newWW,wt):
        WWl.wp = wl
    
    rrz = list()
    gap = list()
    ''' begin loop '''
    for itz in range(10):
        ws = [WWl.wp for WWl in newWW]
        for Q,w in zip(A,wt) :
            Q.changeWeights(w)
        tm = time()
        z = optl1.solveL1(y, A)
        rrz = optl1.rrz # .append(optl1.rrz.pop())
        gap = optl1.gap #.append(optl1.gap.pop())
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved l1 itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()
        
        if plain:
            wmx = [WL.updatePlain(yl, wtl, zl) for WL,yl,wtl,zl in zip(newWW,y,wt,z)]
        else:
            wmx = [WL.updateFourier(yl, wtl, zl) for WL,yl,wtl,zl in zip(newWW,y,wt,z)]
            
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved w update itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()
        wt = [weightAgg(wmxl,p,q,comm) for wmxl in wmx]
        wp = [WL.wp for WL in newWW]
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' have new weights itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        outd = {'y':y, 'z':z, 'wt':wt,'wp':wp,'m':m,'p':p,'q':q, 'rho':rho,'lmb':lmb, 'xi':xi, 'rrz':rrz,'gap':gap, 'ws':ws }
        
        if plain & (dts == 'plmr'):
            spio.savemat('miniOut_' + repr(p) + '_' + repr(nProc) + '_' + repr(rk), outd)
        elif plain & (dts == 'mpk'):
            spio.savemat('miniOutMPK_' + repr(nProc) + '_' + repr(rk), outd)
        else:
            spio.savemat('plainOut_' + repr(nProc) + '_' + repr(rk), outd) 
    
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
            
def weightInit(p,q,comm):
    ''' create some initial, normalized weights '''
    rank = comm.Get_rank()
    if rank == 0:
        wt = np.random.randn(q,p)/np.sqrt(q)
    else:
        wt = None
        
    return comm.bcast(wt,root=0)
        
    
    

def getData(m,dts,ch,rank=0):
    ''' simple function to grab data 
    returns zero mean, normalized data sample
    '''
    
#    import matplotlib.pyplot as plt
    if dts == 'plmr':
        D = spio.loadmat('../data/plmr.mat')
        cix = 1000 + rank*m # np.random.random_integers(0,upb,1)
        slz = slice(cix,cix+m)
        y = D['fs'][slz].astype('complex128').flatten()
        y = y - np.mean(y)

    if dts == 'mpk':
        nbr = (np.floor(rank/2) + 900).astype('int64')
        D = spio.loadmat('../data/lcd' + repr(nbr) + '.mat')
        
        y = list()
        for ix in xrange(ch):
            if rank%2 == 0:
                rng = slice(500000-3*m/4,500000+m/4)
            else:
                rng = slice(500000-m/4,500000+3*m/4)
            yl = D['alldat'][0][ix][rng].astype('complex128').flatten()
            yl = yl-np.mean(yl)
            y.append(yl)
    
    print 'shape of y ' + repr(len(y))
    print [yl.shape for yl in y]
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
    
    
    