
if ~exist('fp')
    fp = load('~/Documents/workspace/sparseConv/src/testout_9_4_0.mat')
end


for ix = 1:101
    rng = (ix-1)*15000 + (1:15000);
    sprs(ix) = sum(abs(fp.z(rng))==0)/15000;
    nrn(ix) = norm(fp.z(rng))/norm(fp.z);
end

figure(1)
subplot(211)
plot(sprs)
title('relative sparsity')


subplot(212)
plot(nrn)
title('relative norm')