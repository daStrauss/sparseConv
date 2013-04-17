
addpath('../common')

for ix = 900:950
    [alldat alltmz] = pull_single(ix, 0);
    
    save(['/home/dstrauss/localDat/lcd' num2str(ix)], 'alldat')
end
