% x = sin(2*pi*5*linspace(0,1,10)').*gausswin(10);
x = randn(30,1).*gausswin(30,10); %  x(1) = 1;
y = randn(100,1);

z = conv(x,y);

bo = 1./(0e-8 + fft([x]));
qo = ifft(bo);


h = conv(qo,z);

P = convmtx(z,30);
R = P(30:129,:);

wp = (R'*R)\(R'*y);

g = conv(wp,z);

figure(13)
plot([x wp qo])

figure(14);
plot(1:100,y, 'o', -29:128,h, -29:128,g, 1:100,R*wp)


% (1:208,g, 

