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


creating some routines to make a convolutional net. especially one that works in parallel

convNetPrime.py is a looped extension of convNet.py
The main goal of this routine is to learn a set of weights, w, such that
the weights can reproduce the data when convolved with a sparse vector. We solve for
both the weights and the sparse vector iteratively, alternating between updates for the weights
and updates for the sparse vector.

The structure of this code allows it to handle multi-channel data. For example, if 3 channels of data
are simultaneously collected, the script looks for a set of sparse vectors with similar sparsity patterns
that convolve with an array of weights, separate weights for each channel, to reproduce the data.

The algorithm is based on consensus optimization for the weights. Independent workers take a parsel of data
solve for a sparse description vector and an appropriate set of weights. Each of these independent workers submits
an opinion about the weights to a central aggregating processor. The aggregating processor finds an average set of
normalized weights close to each of the individual sets. The updated global weights are shipped out and the cycle
begins again.

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
    
    if len(sys.argv) <= 2:
        ''' complain '''
        print 'need  arguments: plain or fft?'
        return 0
    else:
        if sys.argv[1] == 'plain':
            plain = True
        elif sys.argv[1] == 'fft':
            plain = False
        else:
            plain = True
            
    if len(sys.argv) >= 3:
        dts = sys.argv[2]
    else:
        dts = 'plmr'
            
    ''' set the parameters for how big of a problem to grab'''   
    m = 50000 # size of data
    p = 25 # number of filters
    q = 300 # length of filters

            # dts is short for data source 
    if dts == 'plmr':
        rho = 1 # l1 internal parameter
        lmb = 0.005 # l1 weighting
        xi = 0.2 # weights update parameter
        ch = 1 # number of channels
    elif dts == 'mpk':
        rho = 0.1
        lmb = 4e-6
        xi = 0.2
        ch = 3

    ''' load data, given rank '''
    y = getData(m,dts,ch,rank=rk)
    
    ''' initialize weights, looping over the number of channels '''    
    wt = [weightInit(p,q,comm) for ix in xrange(ch)]

    '''create covolution objects of appropriate size for each of the channels '''
    A = [dtm(m,p,q,fourier=not plain) for ix in xrange(ch)]
        
    for Q in A:
        print Q.n

    ''' create a lasso-solver -- this version is designed to accept
    ch channels '''
    optl1 = lasso(m,A[0].n,rho,lmb,ch)

    # update/create new weights    
    newWW = [weightsUpdate.weightsUpdate(m,p,q,xi) for ix in xrange(ch)]
    print newWW[0].m
    for WWl,wl in zip(newWW,wt):
        WWl.wp = wl
    
    rrz = list()
    gap = list()
    '''main learning iterations loop '''
    for itz in range(1000):
        ws = [WWl.wp for WWl in newWW]
        for Q,w in zip(A,wt) :
            # insert new weights into each of the models
            Q.changeWeights(w)
        tm = time()
        # solve for sparse descriptor vectors for all of the channels/models
        z = optl1.solveL1(y, A)
        rrz = optl1.rrz # .append(optl1.rrz.pop())
        gap = optl1.gap #.append(optl1.gap.pop())
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved l1 itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()

        # update the weights 
        if plain:
            wmx = [WL.updatePlain(yl, wtl, zl) for WL,yl,wtl,zl in zip(newWW,y,wt,z)]
        else:
            wmx = [WL.updateFourier(yl, wtl, zl) for WL,yl,wtl,zl in zip(newWW,y,wt,z)]
            
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' solved w update itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        tm = time()
        wt = [weightAgg(wmxl,p,q,comm) for wmxl in wmx]
        wp = [WL.wp for WL in newWW]
        print 'rank ' + repr(rk) + ' of ' + repr(nProc) +  ' have new weights itr: ' + repr(itz) + ' time: ' + repr(time()-tm)
        outd = {'itr':itz, 'y':y, 'z':z, 'wt':wt,'wp':wp,'m':m,'p':p,'q':q, 'rho':rho,'lmb':lmb, 'xi':xi, 'rrz':rrz,'gap':gap, 'ws':ws }
        
        if plain & (dts == 'plmr'):
            spio.savemat('miniOut_' + repr(p) + '_' + repr(nProc) + '_' + repr(rk), outd)
        elif plain & (dts == 'mpk'):
            spio.savemat('miniOutMPK_' + repr(nProc) + '_' + repr(rk), outd)
        else:
            spio.savemat('plainOut_' + repr(nProc) + '_' + repr(rk), outd) 
            
    ''' return these values if I chance to run this in an interactive Python shell'''
    return y,z,optl1,A,wt
    
    
def weightAgg(U,p,q,comm):
    ''' aggregation for weights '''
    # allocate space
    wmx = np.zeros((q,p))

    # combine weights from all of the individual processors
    wmx = comm.allreduce(U,wmx,op=MPI.SUM)

    # compute the mean
    wmx = wmx/comm.Get_size()

    # compute the closest normalized set.
    wt = np.zeros((q,p))
    
    for ix in xrange(p):
        nrm = np.linalg.norm(wmx[:,ix])
        if nrm > 1:
            wt[:,ix] = wmx[:,ix]/nrm
        elif nrm < 0.95:
            wt[:,ix] = 0.95*wmx[:,ix]/(nrm)
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
    This function is "processor aware" that is, when executed
    on an MPI cluster, it will pull a piece of data that has been
    designated for it based on its MPI rank.
    '''
    
#    import matplotlib.pyplot as plt
    if dts == 'plmr':
        D = spio.loadmat('../data/plmr.mat')
        cix = 1000 + rank*m # np.random.random_integers(0,upb,1)
        slz = slice(cix,cix+m)
        y = D['fs'][slz].astype('complex128').flatten()
        y = y - np.mean(y)
        y = [y]
        
    if dts == 'mpk':
        nbr = (np.floor(rank/2) + 900).astype('int64')
        D = spio.loadmat('../data/lcd' + repr(nbr) + '.mat')
        
        y = list()
        for ix in xrange(ch):
            yl = D['alldat'][0][ix][:m].astype('complex128').flatten()
            yl = yl-np.mean(yl)
            y.append(yl)
    
    print 'shape of y ' + repr(len(y))
    print [yl.shape for yl in y]
    return y
    
def testMain():
    ''' I think that this is a script that makes sure that most of the objects function
    properly, namely the convolution objects and their outputs '''
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
    
    
    
