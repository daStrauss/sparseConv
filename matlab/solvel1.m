function s = solvel1(y,u,gmm)
% s = solvel1(y,u)
% solve the lasso problem
% min (y-u*s)_2^2 + gmm*|s|
cvxsltm = tic;
N = size(u,2);
cvx_begin quiet
variable s(N)
minimize( (y-u*s)'*(y-u*s) + gmm*norm(s,1) )
cvx_end



% maybe a polish stage
ind = find(abs(s)>1e-5);
M = u(:,ind);
r = (M'*M)\(M'*y);
s = zeros(size(s));
s(ind) = r;





disp(['solve time ' num2str(toc(cvxsltm))])


% admm stub
      % z = zeros(N,1);
        % x = zeros(N,1);
        % q = zeros(N,1);
        % for itx = 1:400
        %     x = R\(R'\(u'*y(:,g) + rho*(z-q)));
        %     zold = z;
        %     x_hat = alp*x + (1 - alp)*zold;
            
        %     z = svt(x_hat+q,1/rho);
        %     q = q + (x_hat-z);
        %     gpp(itx,g) = norm(z-x_hat);
        %     rr(itx,g) = norm(s-x_hat);
        %     obb(itx,g) = norm(u*z-y(:,g),2)^2 + norm(z,1);
        % end
        

        