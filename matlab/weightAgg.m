function wt = weightAgg(wmx,K,nw)
% simple aggregator across all of the w's
% ensures that the norm doesn't exceed 1.

wv = reshape(mean(wmx,2),nw,K);

for k = 1:K
    nrm = norm(wv(:,k));
    if nrm > 1
        wt(:,k) = wv(:,k)/nrm;
    else
        wt(:,k) = wv(:,k);
    end
    
end


