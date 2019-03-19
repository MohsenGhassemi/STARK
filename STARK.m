function [D_STARK, X_STARK, Reconst_error] = STARK(Y, Permutation_vectors, D_init, param,paramSC)

% Y: observation matrix

% Dictionary_sizes: A 1 by 2 cell containing number of rows and columns of
%factor dictionaries, respectively.

% s: sparsity
% lambda: regularization parameter
% gamma: augmented Lagrangian parameter
% MaxIter_DL: Maximum number of iterations of the DL algorithm
% MaxIter_ADMM: Maximum number of iterations of ADMM
% tol_ADMM: Error toleranec of ADMM
% tol_DL: Reconstrruction (representation) error tolerance of the DL algorithm

%% parameters

Dictionary_sizes=param.DicSizes;
% s=param.Sparsity;
Max_Iter_DL=param.MaxIterDL;
tol_DL=param.TolDL;

MaxIter_ADMM=param.MaxIterADMM;
tol_ADMM=param.TolADMM;
gammaADMM=param.gammaADMM;
lambdaADMM=param.lambdaADMM;

% SparseCodingMethod=param.SparseCodingMethod;
X=[];% Initialization for FISTA and SPAMS

%%
% D_init = normcols(Y(:,randperm(size(Y,2),prod(Dictionary_sizes{2}))));

for  iter = 1:Max_Iter_DL
    
%     % ******      Compressed Sensing Step      ******    
%     if strcmp(SparseCodingMethod,'OMP')
% 
%         X=OMP(D_init,Y,s);
%         
%     elseif strcmp(SparseCodingMethod,'FISTA')
%         ParamFISTA.lambda=param.lambdaFISTA;
%         ParamFISTA.max_iter=param.MaxIterFISTA;
%         ParamFISTA.tol=param.TolFISTA;
%         
%         X = fista_lasso(Y, D_init, X, ParamFISTA);
%         
%     elseif strcmp(SparseCodingMethod,'SPAMS')
%         ParamSPAMS.lambda     = param.lambdaSPAMS;
%         ParamSPAMS.lambda2    = 0;
%         %ParamSPAMS.numThreads = 1;
%         ParamSPAMS.mode       = 2;
%         
%         X = mexLasso(Y, D_init, ParamSPAMS);
%         
%     %elseif strcmp(SparseCodingMethod,'SPARSA')        
%     else
%         disp('Sparse coding is performed by the default method (OMP)')
%         X=OMP(D_init,Y,s);
%     end
    
    X = SparseCoding(Y,D_init,paramSC);
    
    % ******      ADMM (Dictionary Update Step)      ******
    %Reconst_error(iter)=norm(Y-D_init*X,'fro');
    D_STARK = ADMM(Y, X, D_init, Permutation_vectors, Dictionary_sizes,lambdaADMM, gammaADMM, MaxIter_ADMM,tol_ADMM); %D_admm has unit-norm columns
    
    Reconst_error(iter)=norm(Y-D_STARK*X,'fro');
    %Objective_error(iter)=Objective_value_dic_update(end)-Objective_opt;%OMP does not solve the l_1 problem, it enforces the sparsity
    
    if iter>1 && abs(Reconst_error(iter)-Reconst_error(iter-1))<tol_DL
        %    abs(Reconst_error(iter)-Reconst_error(iter-1))
        break
    end
    
    D_init=D_STARK;

end

% X_STARK = OMP(D_STARK, Y,s);
% X_STARK = SparseCoding(Y,D_STARK,paramSC);
X_STARK = X;