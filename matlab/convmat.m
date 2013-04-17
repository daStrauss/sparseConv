
v = randn(100,1);
n = 10;


[mv,nv] = size(v);
v = v(:);		% make v a column vector

%  t = toeplitz([v; zeros(n-1,1)],zeros(n,1));  put Toeplitz code inline
c = [v; zeros(n-1,1)];
r = zeros(n,1);
m = length(c);
x = [r(n:-1:2) ; c(:)];                 % build vector of user data
%
cidx = (0:m-1)';
ridx = n:-1:1;
t = cidx(:,ones(n,1)) + ridx(ones(m,1),:);    % Toeplitz subscripts
t(:) = x(t);                            % actual data
% end of toeplitz code

if mv < nv
	t = t.';
end

