% ok. the point of this script is to know how to update the weights
% properly. need:

p = 5;
q = 10;
m = 100;

wTrue = randn(q,p)/sqrt(q);

z = randn(m*p,1);

sig = applyW(z,wTrue,m);

[sigo M] = applyP(wTrue(:),reshape(z,m,p),q);
b = M'*sig;

save('~/Documents/workspace/sparseConv/src/fakeUP', 'wTrue','z', ...
     'sig','p','q','m','b');






figure(1)
plot([sig sigo])


% wp = prb.wp;
% N = prb.nh;
% K = prb.K;
% wd = prb.wd + (prb.wp-wt);
% nw = prb.nw;


% zp = prb.z((N*K)+(1:N));
% zm = reshape(prb.z(1:(N*K)),N,K);
% fac = sqrt(N/nw/2);

% mdfy = prb.sig - (1/sqrt(N))*(fft(zp))*fac;

% % this rho controls the tightness of 
mdfy = sig;
zm = reshape(z,m,p);

rho = 0.2;
% b = applyPT(mdfy,zm,q); % + rho*(wt(:)-wd(:));

% A = @(x) applyPT(applyP(x,zm,q),zm,q) + rho*x;

b = M'*mdfy;
A = full(M'*M);

[w, FLAG,RELRES,ITER,RESVEC] = pcg(A,b);
wdr = A\b;

bm = applyPT(mdfy,zm,q);

B = @(x) applyPT(applyP(x,zm,q),zm,q);
[wm, FLAG,RELRES,ITER,RESVEC] = pcg(B,bm);

figure(887);
plot([wTrue(:) wdr wm])


% disp(['convg flag ' num2str(FLAG) ' cvg ' num2str(RELRES(end)) ...
%       ' iters ' num2str(ITER) ])


figure(8383); semilogy(RESVEC)

[ag MT] = applyPT(mdfy,zm,q);

figure(101)
plot([b ag])

figure(88)
subplot(121)
imagesc(M')
colorbar

subplot(122)
imagesc(MT)
colorbar


% w = reshape(real(w),nw,K);
% prb.wp = w;
% prb.wd = wd;

% w = w(:)+wd(:);



