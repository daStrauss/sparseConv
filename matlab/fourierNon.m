t = linspace(0,1,1e6)';

fakeData = zeros(size(t));
frq = [5000 7865 2922 10409]*10;
for ix = 1:length(frq)
    fakeData = fakeData + sin(2*pi*frq(ix)*t);
end

% lff = zeros(size(t));

lff = sin(2*pi*frq(1)*t);

sgm = fft(lff(1:128));

