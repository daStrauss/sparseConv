function [patches] = parseTS(data,n,ns);
% simple function to get random patches from the data

N = length(data);
nSamples = ns;

c = ceil(rand(nSamples,1)*(N-n+1));

patches = zeros(n, nSamples);


for ix = 1:nSamples
    patches(:,ix) = data(c(ix) + (0:(n-1)));
end

rawPatches = double(patches);
% pre-whiten

% [u v] = eig(patches*patches'/nSamples);

% patches = u*sqrt(pinv(v))*u'*patches;


% % normalize the data
patches = bsxfun(@minus, patches, mean(patches));

for ix = 1:nSamples
    nn = norm(patches(:,ix));
    patches(:,ix) = patches(:,ix)/nn;
end


% pstd = 3 * std(patches(:));
% scpt = max(min(patches, pstd), -pstd) / pstd;

% % Rescale from [-1,1] to [0.1,0.9]
% scpt = (scpt + 1) * 0.4 + 0.1;

