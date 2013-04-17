function z = applyWT(est,w,N);

K = size(w,2);
n = size(w,1);

% apply w < in backward (i.e. transpose) direction
gtg = floor(n/2) - 1 + (1:N);

for k = 1:K
    intz = conv(est, w(n:-1:1,k));
    z(:,k) = intz(gtg);
end



% size(z)
% z = z(gtg,:);
z = z(:);