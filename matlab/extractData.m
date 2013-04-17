% routine to pull data from the remote server and save it local
% files, mostly because the remote server data base is super
% annoying to work with, and unreliable.
addpath('../common')

for ix = 900:950
    [alldat alltmz] = pull_single(ix, 0);
    
    save(['/home/dstrauss/localDat/lcd' num2str(ix)], 'alldat')
end
