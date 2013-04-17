    
nh = 10000;
nw = 200;
K = 200;
nd = nh+nw-1;

A = @(x) applyW(x,w,nh);
At = @(x) applyWT(x,w,nh);

find_max_eig_func(A,At,nd,K*nh);