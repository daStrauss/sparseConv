function [z M] = applyPT(est,w,N);
% [z M] = applyPT(est,w,N);
% apply transpose operator of the applyP method
% this is convolution matrix transpose operation with more proper bounds.

K = size(w,2);
n = size(w,1);

m = size(est,1);


% apply w < in backward (i.e. transpose) direction
for k = 1:K
    z(:,k) = conv(est, w(n:-1:1,k));
end

% gtg = m - floor(N/2) - 1  + (1:N);

gtg = (m - floor(N/2)):(m+floor(N/2)-1);%  + (1:N);
% size(z)
z = z(gtg,:);
z = z(:);

if nargout > 1
    M = [];
    for k = 1:K
        t = convmtx(w(n:-1:1,k),m);
        M = [M;t(gtg,:)];
    end
end
