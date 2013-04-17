function [est M] = applyP(z,w,N);
% [est M] = applyP(z,w,N);
% I think this is another implementation of the applyW method,
% except that the windowing/'same' settings are slightly different.

K = size(w,2);
z = reshape(z,N,K);
n = size(w,1);

rng = (1:n) + floor(N/2);
% apply w > in forward direction
for k = 1:K
    tmp = conv( z(:,k), w(:,k));
    est(:,k) = tmp(rng);
end

est = sum(est,2);


if nargout > 1
    M = [];
    for k = 1:K
        t = sparse(convmtx(w(:,k), N));
        M = [M t(rng,:)];
    end
end

