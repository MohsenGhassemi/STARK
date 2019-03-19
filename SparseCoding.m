function X = SparseCoding(Y,D,param)
%%%inputs
%param.SparseCodingMethod: 'OMP', 'FISTA', 'SPAMS'
%param.s: sparsity
%param.lambdaFISTA
%param.MaxIterFISTA
%param.TolFISTA
%param.lambdaSPAMS

X=[];
if strcmp(param.SparseCodingMethod,'OMP')
    X=OMP(D,Y,param.s);
    
elseif strcmp(param.SparseCodingMethod,'FISTA')
    ParamFISTA.lambda=param.lambdaFISTA;
    ParamFISTA.max_iter=param.MaxIterFISTA;
    ParamFISTA.tol=param.TolFISTA;
    X = fista_lasso(Y, D, [], ParamFISTA);
    
elseif strcmp(param.SparseCodingMethod,'SPAMS')
    ParamSPAMS.lambda     = param.lambdaSPAMS;
    ParamSPAMS.lambda2    = 0;
    %ParamSPAMS.numThreads = 1;
    ParamSPAMS.mode       = 2;
    X = mexLasso(Y, D, ParamSPAMS);
end

