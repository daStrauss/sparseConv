function M = formAt(w,N);

K = size(w,2);
n = size(w,1);
M = [];
% apply w < in backward (i.e. transpose) direction
rng = floor(n/2) + (1:N);
for k = 1:K
    g = sparse(convmtx( w(:,k), N));
    
    M = [M; g(rng,:)'];
end

% n:-1:1