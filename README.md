sparseConv
==========

Methods for content separation using sparse models

Python extension of the work in the sparseSeparation project

Python development for sparse convolutional models

Python development in parallel with Matlab Development

Licensed under Apache 2.0 license.


The routines developed here in the sparseConv repository are designed to 
implement a method for pursuing 1-norm minimization in problems where the matrix-vector 
multiplication can be described as convolution. Development on this particular repository 
began in December 2012 and was initally parallelized using the classic MPI 
([message passing interface](http://en.wikipedia.org/wiki/Message_Passing_Interface)) 
protocols as adapted in [mpi4py](http://mpi4py.scipy.org/docs/usrman/index.html).

The code developed here was presented in the slides here: ([cvnets](http://www.stanford.edu/~straussd/slides/groupTalks/cvnets.pdf)). The purpose of this code is to short learn wave-like prototypes that can be used to accurately
 explain the observed data in a sparse manner. Learning these weights has been applied to a small
clip of VLF wave data. Methods have been expanded to incorporate multiple simultaneous channels of data.

The learning is parallelized accross different selections of sample data. Independent models fit separate pieces of data 
and come to a consensus on the proper weights for the data set. This level of parallelization requires that we explicitly code
the method for data selection into the MPI jobs. The size of the data instances is limited by the local memory and cpu power of the MPI jobs. Requires careful parameter tuning for reasonable execution times and load balancing.

