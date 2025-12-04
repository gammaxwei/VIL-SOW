function [D,W] = Laplacian_GK(X,k)
% each column is a data
[nFea, nSmp] = size(X);
D = pdist2(X', X', 'Euclidean' );
W = spalloc(nSmp,nSmp,20*nSmp);
[dumb idx] = sort(D, 2); % sort each row
for i = 1 : nSmp
    W(i,idx(i,2:k+1)) = 1;
end
W = (W+W')/2;

D = diag(sum(W,2));
% L = D - W;
