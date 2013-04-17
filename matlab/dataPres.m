g = load('/shared/users/dstrauss/sparseConv/data/plmr.mat')

[spec f t] = spectrogram(g.fs, 1000,500,1024,1e5);
figure(10)
imagesc(t,f/1e3,20*log10(abs(spec)))
hhn = colorbar;
title(hhn,'dB')
xlim([0 100])
xlabel('Time (seconds)')
ylabel('Frequency (kHz)')

set(gca,'YDir', 'normal')
caxis([25 80])

title('Palmer Data ?')

set(gcf,'PaperPosition',[0 0 7 4])
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/lotaPalmer')
