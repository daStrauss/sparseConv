function [est M] = applyW(z,w,N);

K = size(w,2);
z = reshape(z,N,K);

% apply w > in forward direction
for k = 1:K
    est(:,k) = conv( z(:,k), w(:,k),'same');
end

est = sum(est,2);


if nargout > 1
    m = size(w,1);
    rng = floor(m/2) + (1:N);
    
    M = [];
    for k = 1:K
        t = convmtx(w(:,k),N);
        M = [M t(rng,:)];
    end
end
