wTrue = randn(nw,K); % .*repmat(gausswin(10),1,K);

% for n = 1:nSamp
    
%     zTrue = zeros(nh,K);
%     for ix = 1:(K)
%         ndx = round(nh*0.01);
%         zTrue(randi(nh,ndx,1),ix) = ones.*sign(randn(ndx,1));
%         R(:,ix) = conv(zTrue(:,ix), wTrue(:,ix), 'same');
%     end
    
%     zHold(:,n) = zTrue(:);
%     sig(:,n) = sum(R,2) + 0.01*randn(nd,1) + sin(2*pi*400*linspace(0,1,nd)');
% end
    