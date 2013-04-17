function [sig a] = makeRandF(m,k)
% [sig a] = makeRandF(m,k)
% makes a randomly sparse fourier-based signal. neato.
% ensures that the signal is real, but has \pm 1, \pm1i
% coefficients in the fourier domain.

a = zeros(m,1);
n = floor((m-1)/2);
prr = randperm(n);

a(prr(1:k)+1) = 1i*sqrt(m)/2;

a(m:-1:(m-n+1)) = conj(a(1+(1:n)));

sig = real(fft(a))/sqrt(m);