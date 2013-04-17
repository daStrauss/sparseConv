
figure(32);
subplot(211)
stem(prb(1).z)
title('prb.z')

subplot(212)
stem(prb(2).z)
title('prb.z')


figure(31); 
plot([prb(1).sig prb(1).est])

figure(30)
% subplot(121)
imagesc(w)
colorbar
title('w solved')


figure(41)
% subplot(121)
imagesc(log10(abs([prb(:).z])))
colorbar
title('zsolved')

figure(42)
subplot(121)
imagesc([prb(:).sig])
colorbar
title('original')

subplot(122)
imagesc([prb(:).est])
colorbar
title('solved')


figure(20)
% subplot(211)
semilogy(rrz)
title('residuals')

% subplot(212)
% plot(gap)
% title('gaps')


