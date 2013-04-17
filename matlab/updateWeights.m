function [prb w] = updateWeights(prb,wt);
% w = updateWeights(prb,w)
% return a least squares update for the weights- uses cg and
% pipelined operators.

wp = prb.wp;
N = prb.nh;
K = prb.K;
wd = prb.wd + (prb.wp-wt);
nw = prb.nw;


zp = prb.z((N*K)+(1:N));
zm = reshape(prb.z(1:(N*K)),N,K);
fac = sqrt(N/nw/2);

mdfy = prb.sig - (1/sqrt(N))*(fft(zp))*fac;

% this rho controls the tightness of 
rho = 0.2;

b = applyPT(mdfy,zm,nw) + rho*(wt(:)-wd(:));

A = @(x) applyPT(applyP(x,zm,nw),zm,nw) + rho*x;

[w, FLAG,RELRES,ITER,RESVEC] = pcg(A,b);
disp(['convg flag ' num2str(FLAG) ' cvg ' num2str(RELRES(end)) ...
      ' iters ' num2str(ITER) ])


figure(8383); semilogy(RESVEC)


w = reshape(real(w),nw,K);
prb.wp = w;
prb.wd = wd;

w = w(:)+wd(:);


% grd = applyWT(applyW(w(:),zm,nw),zm,nw) - applyWT(prb.sig,zm,nw);
% disp(['time for ista update ' num2str(toc(outUp))])

% ztr = svt(zold + (1/c)*(M'*(prb.sig-M*zold)),g);

% disp(['difference ' num2str(norm(ztr-prb.z))])


% A = [];
% for k = 1:K
%     A = [A sparse(convmtx(zm(:,k),L))];
% end

% gtr = (A'*A)*w(:) - A'*prb.sig; % x;

% disp(['grd diff ' num2str(norm(gtr-grd))])
