function Y = normcols(X)
a = size(X,1);
Y = X./(repmat(sqrt(sum(X.^2)) + 1*~sqrt(sum(X.^2)) ,[a 1]));