
w = randn(10,10);
N = nd;

n = size(w,1);

rng = floor(n/2) + (1:N);
for k = 1
    g = (convmtx( w(:,k), N));
    

end
g = g(rng,:)';

rr = convmtx(w(n:-1:1,1),N);

gt = rr(rng-1,:);
% n:-1:1

figure(10)
imagesc(g)
colorbar

figure(11)
imagesc(g-gt)
colorbar

figure(12)
imagesc(gt)
colorbar
