function X = SparseCoding(Y,D,param)
%%%inputs
%param.SparseCodingMethod: 'OMP', 'FISTA', 'SPAMS'
%param.s: sparsity
%param.lambdaFISTA
%param.MaxIterFISTA
%param.TolFISTA
%param.lambdaSPAMS

if strcmp(param.SparseCodingMethod,'OMP')
    X=OMP(D,Y,param.s);
% Find the code for OMP here: https://github.com/jbhuang0604/FastSC/tree/master/tools/dictionary%20learning/KSVD_Matlab_ToolBox  
% An alternative (faster): http://www.cs.technion.ac.il/~ronrubin/software.html
    
elseif strcmp(param.SparseCodingMethod,'FISTA')
    ParamFISTA.lambda=param.lambdaFISTA;
    ParamFISTA.max_iter=param.MaxIterFISTA;
    ParamFISTA.tol=param.TolFISTA;
    X = fista_lasso(Y, D, [], ParamFISTA);
% Find the code for FISTA here:  https://github.com/tiepvupsu/FISTA    
    
    
elseif strcmp(param.SparseCodingMethod,'SPAMS')
    ParamSPAMS.lambda     = param.lambdaSPAMS;
    ParamSPAMS.lambda2    = 0;
    ParamSPAMS.mode       = 2;
    X = mexLasso(Y, D, ParamSPAMS);
% Find the code for SPAMS here: http://spams-devel.gforge.inria.fr/   
    
    
end

