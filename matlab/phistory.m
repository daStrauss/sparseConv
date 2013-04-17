% base = '~/Documents/workspace/sparseConv/src/miniout_';
% base = '/shared/users/dstrauss/sparseConv/src/miniOutMPK_';
% base = '/shared/users/dstrauss/sparseConv/src/plainOut_';
base = '/shared/users/dstrauss/sparseConv/outp/miniOut_';
% base = '/shared/users/dstrauss/sparseConv/src/miniOutMPK_';


maxWrk = 25;
maxit = 10;
ch = 3
for itr = 1

    for wkr = 1:min(maxWrk,20)
        fp = load([base num2str(maxWrk) ...
                            '_' num2str(wkr-1)]);
        
        offs = (wkr-1)*(itr);
        wh(:,:,itr+offs) = fp.wt;
        
        gaps(:,itr+offs) = fp.gap;
        rrzs(:,itr+offs) = fp.rrz;
        
        za(:,itr+offs) = fp.z;
        

        
        wpwt(itr,wkr) = norm(fp.wt-fp.wp,'fro');
        wnrm(itr,wkr) = mean(sqrt(sum(abs(fp.wt).^2)));
        
        p = double(fp.p);
        m = double(fp.m);
        q = double(fp.q);
        
        tspp(wkr) = sum(abs(fp.z) == 0)/length(fp.z);
        for ix = 1:(p+1)
            rng = (ix-1)*m + (1:m);
            sprs(wkr,ix) = sum(abs(fp.z(rng))==0)/m;
            nrn(wkr,ix) = norm(fp.z(rng))/norm(fp.z);
        end
        
        dta(:,wkr) = real(fp.y);
        est(:,wkr) = real(applyD(fp.z,fp.wp,m));
    end
    
    
    for bkr = 1:itr
        nrm(itr,bkr) = norm(wh(:,:,itr)-wh(:,:,bkr),'fro');
    end
    
end





figure(1)
subplot(211)
imagesc(sprs)
title('relative sparsity')
colorbar


subplot(212)
imagesc(nrn)
title('relative norm')
colorbar

figure(38)
subplot(121)
imagesc(dta)
colorbar
title('data')

subplot(122)
imagesc(est)
colorbar
title('estimates')


figure(8)
subplot(221)
imagesc(wh(:,:,1))
colorbar
title('first iteration weights')

subplot(222)
imagesc(wh(:,:,itr))
colorbar
title('last iteration weights')

[val ind ] = max(sum((wh(:,:,1)-wh(:,:,itr)).^2));

subplot(2,2,3:4)
plot([wh(:,ind,1) wh(:,ind,itr)])
title('vec 1')


figure(10)
subplot(121)
imagesc(gaps)
title('gaps')
colorbar

subplot(122)
imagesc(log10(rrzs))
title('rrzs')
colorbar

figure(11); 
imagesc((abs(za)))
colorbar


