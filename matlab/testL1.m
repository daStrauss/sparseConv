% need to write bonnaFide testing routine for l1 method

p = 5;
q = 10;
m = 100;
k = 15;

wTrue = randn(q,p);
% wn = sqrt(sum(wTrue.^2));
% wTrue = wTrue*diag(1./wn);
[wTrue r] = qr(wTrue, 0);

prm = randperm(m*(p+1));

idxx = prm(1:k);
nf = sum(idxx > m*p);
idxx = idxx(idxx <= m*p);

zTrue = zeros(m*p,1);
zTrue(idxx) = sign(randn(length(idxx),1));

[fsr a] = makeRandF(m,nf);


zTrue = [zTrue; 2*a/sqrt(m)];
[sig M] = applyD(zTrue,wTrue,m);
sig = real(sig) + 0.05*randn(m,1);

[zo MT] = applyDT(sig,wTrue,m);


figure(1)
subplot(211)
plot([sig real(M*zTrue)])

subplot(212)
plot(abs([zTrue zo]))

% [sigo M] = applyP(wTrue(:),reshape(z,m,p),q);
% b = M'*sig;

% % create operators
A = @(x) applyD(x,wTrue,m);
At = @(x) applyDT(x,wTrue,m);


z = zeros(size(zTrue));
zd = zeros(size(zTrue));
zt = zeros(size(zTrue));

rho = 0.1;
g = 0.01;
alpha = 1.8;

rsvk = zeros(31,50);

Atb = At(sig);
Z = @(x) x + (1/rho)*A(At(x));
for iter = 1:50
    b = (Atb + rho*(zd-zt));
    [ss, FLAG,RELRES,ITER,RESVEC] = pcg(Z,A(b),[],30);
    disp([iter ITER])
    rsvk(1:length(RESVEC),iter) = RESVEC;
    
    zn = b/rho - (1/(rho^2))*(At(ss));
    
    zold = zd;
    z = alpha*zn + (1 - alpha)*zold;
    
    zd = svt(z+zt,g/rho);
    
    zt = zt + z-zd;
    rrz(iter) = norm(A(z) - sig);
    gap(iter) = norm(z-zd);
end



% Atb = At(sig);
% Z = @(x) A(At(x));
% for iter = 1:50
%     b = A(zd-zt) - sig;
%     [ss, FLAG,RELRES,ITER,RESVEC] = pcg(Z,b,[],30);
    
%     rsvk(1:length(RESVEC),iter) = RESVEC;
    
%     z = (zd-zt) - At(ss);
    
%     zd = svt(z+zt,1/rho);
    
%     zt = zt + z-zd;
%     rrz(iter) = norm(A(z) - sig);
%     gap(iter) = norm(z-zd);
% end




cvx_begin
variable x(size(z)) complex
minimize(0.5*(sig - M*x)'*(sig-M*x) + g*norm(x,1) )
% subject to
% sig == M*x
cvx_end

%  

figure(2)
plot(abs([zd zTrue x]))

figure(20)
subplot(211)
plot(real([zd zTrue x]))
title('real z')

subplot(212)
plot(imag([zd zTrue x]))
title('imag z')

figure(3)
imagesc(rsvk)
colorbar

figure(4)
subplot(211)
plot(rrz)
title('rrz')

subplot(212)
plot(gap)
title('gap')


save('~/Documents/workspace/sparseConv/src/fakeL1', 'wTrue','zTrue', ...
     'sig','p','q','m','z','zd','zt');

