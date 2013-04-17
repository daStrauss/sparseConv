% script to look at the results from a big python run of the
% 'plain' routine.
base = '/shared/users/dstrauss/sparseConv/src/plainOut_';


maxWrk = 20;
maxit = 10;
ch = 3
for itr = 1

    for wkr = 1:min(maxWrk,20)
        fp(wkr) = load([base num2str(maxWrk) ...
                            '_' num2str(wkr-1)]);
        
        rkp(:,wkr) = fp(wkr).rrz;
    end
    
end

m = double(fp(1).m);
p = double(fp(1).p);
q = double(fp(1).q);

for wkr = 1:maxWrk
    for ch = 1:3
        sprs(wkr,ch) = sum(abs(fp(wkr).z(ch,:))==0)/ ...
            length(fp(wkr).z(ch,:));
        
        for idx = 1:(p+1)
            rng = m*(idx-1) + (1:m);
            spm(wkr,ch,idx) = sum(abs(fp(wkr).z(ch,rng))==0)/m;
        end
        
            
    end
end


figure(2)
imagesc(sprs)
colorbar

figure(3)
for ch = 1:3
    subplot(1,3,ch)
    imagesc(squeeze(spm(:,ch,:)))
    colorbar
end

        

