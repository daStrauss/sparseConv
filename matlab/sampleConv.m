base = '/shared/users/dstrauss/sparseConv/datnow/miniOut_25_';

maxWkr = 20;

wkr = 1;
for wkr = 1
    fp = load([base num2str(maxWkr) '_' num2str(wkr-1)]);
    zzm(:,wkr) = fp.z;
end

m = double(fp.m);
q = double(fp.q);
p = double(fp.p);

wtl = squeeze(fp.wt);

wtl(:,14) = wtl(:,14)*100*6.5;
fp.z((14-1)*m + (1:m)) = 6.5*100*fp.z((14-1)*m + (1:m));


cvl = wtl(:,14);
xx = fp.z((14-1)*m + (1:m));
xx = real(xx(1:50:50000));


figure(100)
plot(cvl)
set(gcf,'PaperPosition', [0 0 4 3]*0.6)
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/smpWiggle')

figure(104)
plot((xx),1:1000)
set(gca,'YDir','reverse')
set(gcf,'PaperPosition', [0 0 1.5 4]*0.6)
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/convVec')


figure(104)
plot(1:1000,(xx))
set(gca,'YDir','reverse')
set(gcf,'PaperPosition', [0 0 4 3]*0.6)
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/convTo')



M = convmtx(cvl,1000);
figure(101)
imagesc(M(151:1151,:))


set(gcf,'PaperPosition', [0 0 4 3]*0.6)
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/wiggleMat')

pp = M(151:1151,:)*xx';
figure(105); 
plot(pp)
xlim([0 1000])

set(gcf,'PaperPosition', [0 0 4 3]*0.6)
print('-depsc2','/shared/users/dstrauss/ant_net/talks/convNets/images/convAns')

