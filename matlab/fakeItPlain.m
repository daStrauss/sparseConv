% a few iterations of the cg methods to compare against what
% happens in the python code

% base = '/shared/users/dstrauss/sparseConv/outp/miniOut_';
% base = '/shared/users/dstrauss/sparseConv/outp/plainOut_';
% base = '/Users/dstrauss/Documents/workspace/sparseConv/src/miniOut_';
% base = '/shared/users/dstrauss/sparseConv/datnow/miniOut_25_';
base = '/shared/users/dstrauss/sparseConv/src/miniOut_25_';

maxWkr = 20;

wkr = 1;
for wkr = 1
    fp = load([base num2str(maxWkr) '_' num2str(wkr-1)]);
    zzm(:,wkr) = fp.z;
end

m = double(fp.m);
q = double(fp.q);
p = double(fp.p);

% 1-5
% 6-10
% 11 - 15
% 16 - 20

wtl = squeeze(fp.wt);

% wtl(:,14) = wtl(:,14)*100*6.5;
% fp.z((14-1)*m + (1:m)) = 6.5*100*fp.z((14-1)*m + (1:m));

gap = [0.02 0.02];
marg_h = [0.1 0.1];
marg_w = [0.1 0.1];
Nh = 5;
Nw = 5;

axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
axwl = ((1-sum(marg_w)-(Nw-1)*gap(2))/Nw)*4 + 3*gap(2);
axwr = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;

axwf = (1-sum(marg_w));

py = 1-marg_h(2)-axh;
figure(1); clf
for y = 1:Nh
    px = marg_w(1);
    %for x = 1:2
        innx = 1+(y-1)*2;
        % wide windows
        ha(innx) = axes('Units','normalized', ...
                        'Position', [px py axwl axh], ...
                        'XTickLabel','',...
                        'YTickLabel','');
        innx = 2+(y-1)*2;
        px = px + axwl + gap(2);
        ha(innx) = axes('Units','normalized', ...
                        'Position', [px py axwr axh], ...
                        'XTickLabel','',...
                        'YTickLabel','');
        py = py - axh - gap(1); 
end



% hgg = tight_subplot(4,5,[.02 .02],[.1 .1],[.1 .1])

gox = [8 9 23 24];

% subplot(4,5,1:4)
for pp = 1:4
    axes(ha((pp-1)*2+1))
    plot(real(fp.z((gox(pp)-1)*m + (1:m))))
    set(gca,'XTickLabel', [])
    set(gca,'YTickLabel', [])
    ylim([-200 200])
    text(0.25e4,60,['h_' num2str(pp)],'BackgroundColor','w','EdgeColor','k')

    axes(ha((pp-1)*2+2))
    plot(real(wtl(:,gox(pp))))
    xlim([0 300])
    ylim([-0.2 0.2])
    set(gca,'YTickLabel', [])
    set(gca,'XTickLabel', [])
    % text(50,0,['w_' num2str(pp)],'BackgroundColor','w','EdgeColor','k')
end


% px = marg_w(1);
% py = 1-marg_h(2)- 4*axh - 3*gap(1); 

% ha(innx+1) = axes('Units','normalized', ...
%                   'Position', [px py axwf axh], ...
%                   'XTickLabel','',...
%                   'YTickLabel','');

% set(gca,'XTickLabel', [])


% px = marg_w(1);
% py = 1-marg_h(2)- 5*axh - 4*gap(1); 

% ha(innx+2) = axes('Units','normalized', ...
%                   'Position', [px py axwf axh], ...
%                   'XTickLabel','',...
%                   'YTickLabel','');

        

% axes(ha(7))
% plot(abs(fp.z(p*m+(1:m))))
% set(gca,'XTickLabel', [])
% set(gca,'YTickLabel', [])
% ylim([0 200])
% text(0.25e4,125,'Fourier','BackgroundColor','w','EdgeColor','k')


% axes(ha(8))
% set(gca,'Visible','off')


axes(ha(9))

plot(real(fp.y)/1e4)
ylim([-1 1])
text(0.25e4,0,'Data','BackgroundColor','w','EdgeColor','k')
set(gca,'YTickLabel',[])

axes(ha(10))
set(gca,'visible','off')


set(gcf,'PaperPosition',[0 0 8 5]*0.8)
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/addSumPlain')


mkk = reshape(squeeze(fp.z), m,p);

figure(3);
imagesc(abs(mkk))
colorbar

figure(4);
subplot(131)
imagesc(squeeze(fp.wt))
colorbar

subplot(132)
imagesc(squeeze(fp.wp))
colorbar

subplot(133)
imagesc(squeeze(fp.ws))
colorbar



figure(5);clf
h = tight_subplot(2,2,[0.03 0.03],[0.1 0.1],[0.1 0.1]);

for ixl = 1:4
    axes(h(ixl))
    plot(real(wtl(:,gox(ixl))))
    xlim([0 300])
    ylim([-0.2 0.2])
    set(gca,'YTickLabel', [])
    set(gca,'XTickLabel', [])
end

set(gcf,'PaperPosition', [0 0 8 4])
print('-depsc2', '/shared/users/dstrauss/ant_net/talks/convNets/images/sampleWeights')

