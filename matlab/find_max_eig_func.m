function lmax = find_max_eig_func(A,At,n,N)
% function lmax = find_max_eig(A)
% calculates the maximum eigenvalue for At*A 
% where A,At are function handles for applying A and A' repsectively
% A*A where A is some sort of
% matrix or spot operator
% [n N] = size(A);

% procedure to calculate maximum eigenvalue - to determine c
egvt = tic;
z = randn(N,1);
p = z/norm(z);
for iter = 1:100
   z = At(A(p));
   p = z/norm(z);
   lm(iter) = norm(z)/norm(p);
   if (iter>=2) && (abs(lm(iter)-lm(iter-1))/(lm(iter)) < 1e-5)
       break
   end
end
lmax = lm(iter);
disp(['Found max eigenvalue ' num2str(lmax) ' itrs ' ...
      num2str(iter) ' final diff ' num2str(abs(lm(iter)-lm(iter-1))) ...
     ' in time ' num2str(toc(egvt))])

% figure(383);
% plot(lm)