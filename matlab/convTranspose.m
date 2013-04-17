% testing routine for the applyW/applyWT methodology.
n = 10;
N = 100;
x = randn(N,1);
y = randn(n,1);


zc = conv(x,y);
zf = applyW(x,y,N);
ztg = conv(x,y,'same');

% tic
% CM = convmtx(y,N);

% zm = CM*x;
% xbk = CM'*(CM*x);
% toc


tic
xf = applyWT(applyW(x,y,N),y,N);
toc

xcc = conv(zc,y(n:-1:1));


% figure(2)
% plot(1:N,xbk,'o',1:N, conv(ztg,y(n:-1:1),'same'), -(n-2):(N+n-1),xcc, 1:N,xf)


f = @(x) applyW(x,y,N);
ft = @(x) applyWT(x,y,N);

tic
ll = find_max_eig_func(f,ft,N,N);
toc


K = 10;
wTrue = randn(n,K);
zTrue = zeros(N,K);
for ix = 1:(K)
    nzro = round(N*0.05);
    zTrue(randi(N,nzro,1),ix) = ones.*sign(randn(nzro,1));
    R(:,ix) = conv(wTrue(:,ix),zTrue(:,ix));
end

zHold(:,n) = zTrue(:);
sig = sum(R,2);

M = [];
for k = 1:K
    M = [M sparse(convmtx(wTrue(:,k),N))];
end

sgf = applyW(zTrue(:),wTrue,N);

sgm = M*zTrue(:);

figure(20);
plot(1:(N+n-1),sig, 'o', 1:(N+n-1),sgf)

tic
ztx = applyWT(sgf,wTrue,N);
toc
tic
zmm = M'*sgf;
toc
figure(21)
plot([zmm ztx])