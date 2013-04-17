% looks like a comparison between a few different algorithms for l1
% separation a CVX method and a non-cvx method. But this is mostly
% a method for comparing over randomly generated data.
clear all
N = 50;
ne = 150;

A = randn(N);
[Q R] = qr(A);

for g = 1:150
    k = ceil(rand*8);
    p = randperm(N);
    x = zeros(N,1);
    x(p(1:k)) = randn(k,1);
    
    y(:,g) = Q*x;
    tS(:,g) = x;
end


[u s v] = svd(y);

W = u;

gmm = .5;
rho = 1000;
alp = 1.5;

for itt = 1:100
    globTm = tic;
    
    parfor g = 1:150
        S(:,g) = solvel1(y(:,g),W,gmm);
    end

    
    cvx_begin
    variable W(N,N)
    minimize( norm(y-W*S,'fro') )
    subject to
    sum(W.^2,2) <= 1
    cvx_end
    disp(['full it time = ' num2str(toc(globTm))])
    
    dimin(itt) = norm(y-W*S,'fro');
end
