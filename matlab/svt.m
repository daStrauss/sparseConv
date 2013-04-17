function u = svt(x,tt);
% soft thresholding 
u = max(1-tt./abs(x),0).*x;
