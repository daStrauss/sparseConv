function [sig a] = makeRandF(m,k)

a = zeros(m,1);
n = floor((m-1)/2);
prr = randperm(n);

a(prr(1:k)+1) = 1i*sqrt(m)/2;

a(m:-1:(m-n+1)) = conj(a(1+(1:n)));

sig = real(fft(a))/sqrt(m);