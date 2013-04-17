function [est M] = applyW(z,w,N);
% [est M] = applyW(z,w,N);
% first pass at implicit matrix-vector multiplication instantiated
% by convolution
% z is the data to transform, w, is an array of weights, N is the
% dimensino of the output data.

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
