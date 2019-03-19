function [D_STARK, X_STARK, Reconst_error] = STARK(Y, Permutation_vectors, D_init, paramDL, paramSC)

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

Dictionary_sizes=paramDL.DicSizes;
Max_Iter_DL=paramDL.MaxIterDL;
tol_DL=paramDL.TolDL;
MaxIter_ADMM=paramDL.MaxIterADMM;
tol_ADMM=paramDL.TolADMM;
gammaADMM=paramDL.gammaADMM;
lambdaADMM=paramDL.lambdaADMM;


for  iter = 1:Max_Iter_DL
    
    % ******      Compressed Sensing Step      ******    
    
    X_STARK = SparseCoding(Y,D_init,paramSC);
    
    
    % ******      ADMM (Dictionary Update Step)      ******
    
    D_STARK = ADMM(Y, X, D_init, Permutation_vectors, Dictionary_sizes,lambdaADMM, gammaADMM, MaxIter_ADMM,tol_ADMM); 
    %D_admm has unit-norm columns
    
    Reconst_error(iter)=norm(Y-D_STARK*X,'fro');
    
    if iter>1 && abs(Reconst_error(iter)-Reconst_error(iter-1))<tol_DL
        break
    end
    
    D_init=D_STARK;

end
