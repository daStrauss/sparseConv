% do the designerConvolution method as a local matlab
% script.
% best for comparing with respect to python.
% addpath starter

% M = load('/shared/users/dstrauss/ant_net/multiArm/local1.mat');

% if matlabpool('size') < 6
%     matlabpool open
% end


nSamp = 2; % number of samples
nw = 300; % size of patches
nh = 5000; % size of sparse vector

K = 50; % number of independent patches

% initialize some weights

load('~/Documents/workspace/sparseConv/src/fakew.mat') 
% w = 0.05*randn(nw,K);
w = wini;
wt = w;

% for n = 1:nSamp
%     M = load(['/home/dstrauss/localDat/lcd' num2str(n+899)]);
%     sig(:,n) = M.alldat{1}(1:nd);
% end

M = load('~/Documents/workspace/sparseConv/data/plmr');

for n = 1:nSamp
    sig(:,n) = M.fs(1000 + (1:nh) + (n-1)*nh);
    sig(:,n) = sig(:,n)-mean(sig(:,n));
    sig(:,n) = sig(:,n)/norm(sig(:,n));
end

% sig = sig.*repmat(msk,1,nSamp);

disp('loaded data')

prb = repmat(struct('sig', [], ...
                    'est', zeros(nh,1), ...
                    'thrsh', [], ...
                    'z',zeros(nh*(K+1), 1), ...
                    'wp', w, ...
                    'wd', zeros(size(w)),...
                    'nh', nh, ...
                    'nw', nw,...
                    'K', K),1,nSamp);

disp('built structures')
for n = 1:nSamp
    prb(n).sig = sig(:,n);
end

disp('here we go!')



for iter = 1:20
    glbl = tic;
    
    for n = 1:nSamp
        [prb(n) rrz(:,iter,n)] = paraZup(prb(n),5e-4);
        rlsp(iter,n) = sum(abs(prb(n).z) ~= 0)/length(prb(n).z);
        [prb(n) wmx(:,n)] = updateWeights(prb(n),wt);
        
    end
    
    
    wt = weightAgg(wmx,K,nw);
    
    
    
    disp(['iter ' num2str(iter) ' time ' num2str(toc(glbl)) ...
          ' rel sparse ' num2str(rlsp(iter))])
    
end
    
% save('realPalmrv5', 'prb', 'w','wini','wt', '-v7.3')


figure(32);
subplot(211)
plot(real(prb(1).z))
title('real final prb.z')


subplot(212)
plot(imag(prb(1).z))
title('real final prb.z')


figure(31); 
subplot(211)
plot(real([prb(1).sig prb(1).est]))
title('real final - sig&est')

subplot(212)
plot(imag([prb(1).sig prb(1).est]))
title('imag final - sig&est')

figure(30)
subplot(121)
imagesc(wt)
colorbar
title('w solved')

subplot(122)
imagesc(wini)
title('w True')
colorbar

% figure(41)
% subplot(131)
% imagesc(zHold)
% colorbar
% title('ztrue')

% subplot(132)
% imagesc([prb(:).z])
% colorbar
% title('zsolved')

% subplot(133)
% imagesc([prb(:).zt])
% colorbar
% % title('z t')

% figure(42)
% subplot(121)
% imagesc(sig)
% colorbar
% title('original')

% subplot(122)
% imagesc([prb(:).est])
% colorbar
% title('solved')


figure(20)
% subplot(211)
semilogy(rrz)
title('final residuals')

% subplot(212)
% plot(gap)
% title('gaps')


