function [est M] = applyD(z,w,N);
% [est M] = applyD(z,w,N);
% Apply the convolution matrix using the weight array w, to the
% vector z. uses data vector of lenth N.

K = size(w,2);
zp = z((N*K)+(1:(N)));
z = reshape(z(1:(N*K)),N,K);
nw = size(w,1);
fac = sqrt(N/nw/2);

% apply w > in forward direction
for k = 1:K
    est(:,k) = conv(z(:,k), w(:,k),'same');
end


est = sum(est,2);
est = est + (fft(zp))*(1/sqrt(N));


if nargout > 1
    m = size(w,1);
    rng = floor(m/2) + (1:N);
    
    M = [];
    for k = 1:K
        t = convmtx(w(:,k),N);
        M = [M t(rng,:)];
    end
    M = [M (1/sqrt(N))*fft(eye(N))];
end
