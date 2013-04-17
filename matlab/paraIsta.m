function [prb, rrz, dz] = paraIsta(prb,K,w,nh,nw,c,g)
% [Q, x, prb] = paraZup(prb,K,w,lData,f,M,g,r)
% where N is the length of the "sparse vector window"

% finv = @(x) f.T*(f.U\(f.L\(f.S*(f.R\x))));
outUp = tic;

msk = conv(gausswin(K),ones(nh,1));

A = @(x) applyD(x,w,nh);
At = @(x) applyDT(x,w,nh);

% c = 1.2*find_max_eig_func(A,At,N+L-1,K*N);
disp(['c = ' num2str(c)])
zold = prb.z;
z = zeros(size(prb.z));
est = zeros(size(prb.sig));

for q = 1
    
    dz = (1/c)*(At((prb.sig-est)));
    tth = 2*median(abs(dz));
    z = svt(z + dz,tth);
    est = A(z);
    
    rrz(q) = norm(prb.sig-est);
    prb.thrsh(q) = tth;
end

prb.est = est;
prb.z = z;

% zm = reshape(prb.z, nh,K);

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

grd = 0;