function Y = shrink(X,threshold)
[U,S,V] = svd(X,'econ');
S(S<=threshold) = 0;
S(S>threshold) = S(S>threshold) - threshold;
Y = U*S*V';