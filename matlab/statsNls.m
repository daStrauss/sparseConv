nSamp = 20; % number of samples
nw = 50; % size of patches
nh = 10000; % size of sparse vector

K = 100; % number of independent patches


for ix = 1:nSamp
    zm = reshape(prb(ix).z,nh,K+1);
    crdcnt(:,ix) = sum(abs(zm)~=0);
end


for ix = 1:nSamp
    nrwp(ix) = norm(prb(ix).wp -wt,'fro');
end
