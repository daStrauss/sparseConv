function [prb, rrz] = paraZup(prb,g)
% [prb, rrz] = paraZup(prb,K,w,nh,nw,g)

K = prb.K;
w = prb.wp;
nh = prb.nh;
nw = prb.nw;

% create operators
A = @(x) applyD(x,w,nh);
At = @(x) applyDT(x,w,nh);



z = prb.z;
zd = zeros(size(z));
zt = zeros(size(z));

rho = 5;

% M = @(x) At(A(x)) + rho*x;
M = @(x) x + (1/rho)*A(At(x));

n = size(prb.sig,1);

Atb = At(prb.sig);
przt = tic;

for iter = 1:20
    b = Atb + rho*(zd-zt);
    [ss, FLAG,RELRES,ITER,RESVEC] = pcg(M,A(b));
    
    % figure(383)
    % subplot(211)
    % plot(ss)
    
    % subplot(212)
    % plot(1:1000, M(ss),1:1000,A(b))
    % sap = Q\(A(b));
    
    % [z, FLAG,RELRES,ITER,RESVEC] = pcg(M,b);
    disp(['inner iter ' num2str(iter) ' ' num2str(FLAG) ' ' ...
          num2str(RELRES(end)) ' iters ' num2str(ITER)])
    % figure(200); semilogy(RESVEC)
    % title('parazup')
    
    % disp(['norm d ' num2str(norm(sap-ss))])
    z = b/rho - (1/(rho^2))*(At(ss));
    
    zd = svt(z+zt,g/rho);
    
    zt = zt + z-zd;
    rrz(iter) = norm(A(z) - prb.sig);
    gap(iter) = norm(z-zd);
end

disp(['time for prz ' num2str(toc(przt))])
prb.z = zd;

prb.thrsh = 0;
prb.est = A(z);

% figure(300)
% subplot(211)
% plot(rrz)
% title('rrz')

% subplot(212)
% plot(gap)
% title('gap')

% figure(883);
% plot([abs(ss)])