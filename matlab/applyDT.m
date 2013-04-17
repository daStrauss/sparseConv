function [z M] = applyDT(est,w,N);
% [z M] = applyDT(est,w,N);
% applies transpose operator of convolutions and fourier expansion
% transpose of applyD
K = size(w,2);
n = size(w,1);

nw = size(w,1);
fac = sqrt(N/nw/2);

gtg = floor(n/2) - 1 + (1:N);

% apply w < in backward (i.e. transpose) direction
for k = 1:K
    intz = conv(est, w(n:-1:1,k));
    z(:,k) = intz(gtg);
end
zp = ifft(est)*sqrt(length(est));

% gtg = (n-1) + (1:N);
% size(z)
% z = z(gtg,:);
z = z(:);
z = [z;zp];

if nargout > 1

    
    M = [];
    for k = 1:K
        t = convmtx(w(:,k),N);
        M = [M; t(gtg,:)];
    end
    M = [M ;(sqrt(N))*ifft(eye(N))];
end


