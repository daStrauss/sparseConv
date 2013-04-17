function M = formA(w,N);

M = [];

K = size(w,2);
n = size(w,1);

% apply w > in forward direction
rng = floor(n/2) + (1:N);

for k = 1:K
    g = sparse(convmtx( w(:,k), N));
    
    M = [M g(rng,:)];
end
