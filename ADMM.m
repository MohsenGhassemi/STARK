function D=ADMM(Y, X, D_init, Permutation_vectors,Dictionary_sizes,lambda, gamma, MaxIter,tol)

% Y: observation matrix
% X: coefficient matrix
% D_init: initial dictionary

% Dictionary_sizes: A 1 by 2 cell containing number of rows and columns of
%factor dictionaries, respectively.

% K: tensor order
% lambda: regularization parameter
% gamma: augmented Lagrangian parameter
% MaxIter: Maximum number of iterations


m = Dictionary_sizes{1};
p=  Dictionary_sizes{2};
K=length(m);
size_dpi = fliplr(m.*p);% [m(3)*p(3) m(2)*p(2) m(1)*p(1)];


% Dictionary Initialization

D_pi_init = rearrange_D_Dpi(D_init,m ,p);


% Dummy variables and Lagrange multipliers initialization
A=cell(1,K);
W=cell(1,K);

for k=1:K
    A{k}= rand(size_dpi);
    W{k}=D_pi_init;
end
%


%% % perform the ADMM algorithm

PvT=Permutation_vectors(:,2); 
unvec_1=Y*X';
unperm_vec_1=unvec_1(:);
vec_1=unperm_vec_1(PvT);

% (X kron I)(X kron I)^T=(XX^T) kron I
X_Gram=X*X';
X_tilde_Gram=sparse(kron(X_Gram,speye(prod(m))));
matr = X_tilde_Gram(PvT,PvT) + gamma*K*speye(prod(m)*prod(p));


for iter=1:MaxIter
    sum_A_W=zeros(size_dpi);
    
    for k=1:K
        sum_A_W=sum_A_W+A{k}+gamma*W{k};
    end
    
    
    D_pi = reshape(matr\(vec_1 + sum_A_W(:)) ,size_dpi);
    
    for k=1:K
        
        % update W_n equation(32)
        W{k} = unflatten(shrink(flatten(D_pi - A{k}/gamma, k), lambda/(gamma)), k, size_dpi);
        
        % update the Lagrange multiplier equation(33)
        A{k} = A{k} - gamma*(D_pi - W{k});
    end
    
    
end

D = normcols(rearrange_Dpi_D(D_pi,m,p));


