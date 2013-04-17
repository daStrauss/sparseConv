
N = 1000;

n = 50;

sig(:,1) = [randn(25,1); zeros(25,1)];
sig(:,2) = [randn(30,1); zeros(20,1)];
sig(:,3) = [randn(45,1); zeros( 5,1)];

test = sin((1:1000)'*2*pi*0.02);

p = randperm(N);

for r = 1:20
    

figure(1);
plot(test)