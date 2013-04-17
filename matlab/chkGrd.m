A = randn(15,10);

xTrue = randn(10,1);

b = A*xTrue;


% M = [speye(15) A'; A sparse(10,10)];
% sln = M\[zeros(15,1); b];

sln = (A'*A)\(A'*b);

% xg = 1e-8*randn(10,1);

xg = sln; xg(1) = xg(1) + 0.05;
xg = randn(10,1);
alp = 0.005;

for itx = 1:200
    xg = xg - alp*(2*A'*(A*xg-b) );
    ff(itx) = norm(A*xg - b,2)^2;
    rr(itx) = norm(sln-xg);
end
