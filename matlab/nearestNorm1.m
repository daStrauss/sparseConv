
n = 33;
z = randn(100,n);



cvx_begin
variable x(100)
minimize( norm(z-repmat(x,1,n),'fro')  )
subject to
x'*x <= 1

cvx_end

figure(9); 

plot([z x])

