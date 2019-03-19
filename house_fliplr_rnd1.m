% Y: observation matrix
% D: dictionary
% X: coefficient matrix
% lambda: regularization term
% K: tensor order
% gamma: Augmented Lagrangian multiplier
% Objective function
%F_D = (1/2)*norm(Y-D*X,'fro')^2 + (lambda/N)*sum([unfold(W_1,1),unfold(W_2,2),unfold(W_3,3)]);
clc; clear; close all
load('Data/rand_state1.mat');

addpath(genpath('PARAFAC'));
addpath(genpath('FISTA-SPAMS'));
%addpath(genpath('SPARSA'));
% addpath(genpath('spams-matlab-v2.6'));
rng(rand_state1);
N_montcarl =8;
disp('rand_state1_8montecarlo')
%% Loading image and extracting overlapping patches from it
%load('landscape.mat')
%Image=photons;
Image=double(imread('Data/house_color.tiff'));

input_data = double(Image)/max(max(max(Image)));
[a,b,c]=ind2sub([size(input_data,1),size(input_data,2),size(input_data,3)],find(Image));
input_data=input_data(min(a):max(a),min(b):max(b),:);

dim1=size(input_data,1);
dim2=size(input_data,2);

N_freq = 3;% # of frequencies (# of features in the 3rd mode)
patch_size = 8;

%cropping the image for perfect tiling with no extra pixels
input_data=input_data(1:patch_size*floor(dim1/patch_size),1:patch_size*floor(dim2/patch_size),1:N_freq);
N_pixels=size(input_data,1)*size(input_data,1);

%new (cropped) dimensions
dim1=size(input_data,1);
dim2=size(input_data,2);

%Converting data into patches
%block_data = mat2cell(input_data,patch_size*ones(1,dim1/patch_size),patch_size*ones(1,dim2/patch_size),N_freq);%two dimensional cell array,
% each cell contains a (patch_size*patch_size*N_freq) data point

step=2; %Stride
for i=0:step:dim1-patch_size% step determines how much of the neighboring patches overlap
    for j=0:step:dim2-patch_size
        block_data{i/step+1,j/step+1}=input_data(i+1:i+patch_size, j+1:j+patch_size,:);
        % block_data_indices{i/step+1,j/step+1}=[i+1:i+patch_size; j+1:j+patch_size];
    end
end

%cropping the imageto exclude the extra pixels (due to step size)
input_data=input_data(1:i+patch_size,1:j+patch_size,1:N_freq);

dim1_block=size(block_data,1);
dim2_block=size(block_data,2);

N_blocks =dim1_block*dim2_block ;%number of blocks (data points)

obsrvtn_tensor = zeros(patch_size,patch_size,N_freq,N_blocks);
obsrvtn_vect = zeros(patch_size^2*N_freq,N_blocks);

k=0;
for i = 1:size(block_data,1)
    for j = 1:size(block_data,2)
        k = k+1;
        obsrvtn_tensor(:,:,:,k) = block_data{i,j};
        %        patches_indices(:,:,k)=  block_data_indices{i,j};% the indices of the pixels that are in each block
        obsrvtn_vect(:,k) = reshape(obsrvtn_tensor(:,:,:,k),patch_size^2*N_freq,1,1);%each point is vectorized
    end
end
%%% Noisy and noiseless data
Y_clean = obsrvtn_vect; %clean data
%Y_clean_tns = obsrvtn_tensor(:,:,:,rand_perm);

%% Dictionary Parameters
M = [patch_size, patch_size, N_freq];
P =[2*patch_size ,2*patch_size, N_freq];

Dictionary_sizes{1}=fliplr(M);
Dictionary_sizes{2}=fliplr(P);%needs to be flipped

m=prod(M);
p=prod(P);

%% Experiment setup
sample_sizes=length(Y_clean);%1000*[0.5 1 2 5 10]%[1 5 10 15]
length_sample_sizes=length(sample_sizes);

N_sample = length(sample_sizes);
N_sigs = 3;

Rep_err_train_KSVD = zeros(N_montcarl,N_sigs);
Rep_err_train_LS = zeros(N_montcarl,N_sigs);
Rep_err_train_sum_LS = zeros(N_montcarl,N_sigs);
Rep_err_train_SEDIL = zeros(N_montcarl,N_sigs);
Rep_err_train_STARK = zeros(N_montcarl,N_sigs);
Rep_err_train_TeFDiL = zeros(N_montcarl,N_sigs);
Rep_err_train_TeFDiL2 = zeros(N_montcarl,N_sigs);
Rep_err_train_TeFDiL32 = zeros(N_montcarl,N_sigs);

Rep_err_test_KSVD = zeros(N_montcarl,N_sigs);
Rep_err_test_LS = zeros(N_montcarl,N_sigs);
Rep_err_test_sum_LS = zeros(N_montcarl,N_sigs);
Rep_err_test_SEDIL = zeros(N_montcarl,N_sigs);
Rep_err_test_STARK = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL32 = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL2 = zeros(N_montcarl,N_sigs);

Rep_err_test_KSVD_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_LS_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_sum_LS_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_SEDIL_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_STARK_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL32_OMP = zeros(N_montcarl,N_sigs);
Rep_err_test_TeFDiL2_OMP = zeros(N_montcarl,N_sigs);


[Permutation_vector, Permutation_vectorT]=permutation_vec(Dictionary_sizes);
%Permutation_vector: vector containing the index of non-zero value of the permutation matrix in each row
%Permutation_vectorT: contains those of the transpose of the permutation matrix
Permutation_vectors=[Permutation_vector, Permutation_vectorT];

%% Algorithm Parameters
K = 3; %tensor order
% Sparse Coding Parameters. Have to select sparse coding method here:
% 'OMP', 'SPAMS', 'FISTA'
% paramSC is input to all DL algorithms
s = ceil(p/20); %sparsity level
paramSC.s = s;
paramSC.lambdaFISTA = .1;
paramSC.MaxIterFISTA = 10;
paramSC.TolFISTA = 1e-6;
paramSC.lambdaSPAMS = 1;
paramSC.SparseCodingMethod= 'FISTA';

% Dictionary Learning Parameters
Max_Iter_DL = 50;

% KSVD Parameters
ParamKSVD.L = s;   % number of elements in each linear combination.
ParamKSVD.K = p; % number of dictionary elements
ParamKSVD.numIteration = Max_Iter_DL; % number of iteration to execute the K-SVD algorithm.
ParamKSVD.memusage = 'high';
ParamKSVD.exact = 1;
ParamKSVD.errorFlag = 0;
ParamKSVD.preserveDCAtom = 0;
ParamKSVD.displayProgress = 0;


% STARK Parameters
ParamSTARK.TolADMM = 1e-4; %tolerance in ADMM update
ParamSTARK.MaxIterADMM = 10;
ParamSTARK.DicSizes=Dictionary_sizes;
ParamSTARK.Sparsity=s;
ParamSTARK.MaxIterDL=Max_Iter_DL;
ParamSTARK.TolDL=10^(-4);

% TeFDiL Parameters
ParamTeFDiL.MaxIterCP=50;
ParamTeFDiL.TensorRank=1;
ParamTeFDiL.DicSizes=Dictionary_sizes;
ParamTeFDiL.Sparsity=s;
ParamTeFDiL.MaxIterDL=Max_Iter_DL;
ParamTeFDiL.TolDL=10^(-4);
ParamTeFDiL.epsilon=0.01;%to impprove the condition number of XX^T. multiplied by its frobenious norm.

sig_vals = [10 50];
for mont_crl = 1:N_montcarl
    mont_crl
    for sigma_ind=1:2
        sigma_ind
        sigma = sig_vals(sigma_ind);
        Y_noisy=Y_clean+sigma/max(max(max(Image)))*randn(size(Y_clean)); % noisy data
        
        for  n_cnt= 1:length_sample_sizes
            %Generate training data
            N=sample_sizes(n_cnt);
            % noisy training data
            Y_train = Y_noisy(:,randperm(N_blocks,N));
            
            %iinitializing coordinate dictionaries
            D_init_k={1,3};
            
            Y_tns = reshape(Y_train,M(1),M(2),M(3),N);
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                cols_k = randperm(N*prod(M)/M(k),P(k));
                D_init_k{k} = normcols(D_initk(:,cols_k));
            end
            D_init = kron(D_init_k{3},kron(D_init_k{2},D_init_k{1}));
            %% KSVD with OMP
%             disp('Training unstructured dictionary using K-SVD')
%             if N >= p% does not work for N<p
%                 %                     paramet(mont_crl).initialDictionary = D_init;
%                 ParamKSVD.InitializationMethod = 'DataElements';
%                 [D_KSVD,output] = KSVD(Y_train,ParamKSVD,paramSC);
%                 %%%Dictionary training Representation Error
%                 Rep_err_train_KSVD(mont_crl,sigma_ind) = norm(Y_train - D_KSVD*output.CoefMatrix,'fro')^2/numel(Y_train);
%                 %%%Dictionary test Representation Error
%                 X_test_KSVD = SparseCoding(Y_noisy,D_KSVD,paramSC);
%                 Y_KSVD = D_KSVD*X_test_KSVD;
%                 [Image_out_KSVD, cnt_KSVD] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_KSVD,N_freq);
%                 Rep_err_test_KSVD(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_KSVD./cnt_KSVD,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%                 %with OMP
%                 X_test_KSVD_OMP = OMP(D_KSVD,Y_noisy,s);
%                 Y_KSVD_OMP = D_KSVD*X_test_KSVD_OMP;
%                 [Image_out_KSVD_OMP, cnt_KSVD] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_KSVD_OMP,N_freq);
%                 Rep_err_test_KSVD_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_KSVD_OMP./cnt_KSVD,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%                 
%             else
%                 disp('Insufficient number of training samples for K-SVD')
%             end
            %% KHOSVD with OMP
            %             disp('Training structured dictionary using K-HOSVD')
            %             %         [D_KHOSVD,X_KHOSVD] = khosvd_3D(Max_Iter_DL, Y_train, D_init, M(1), M(2), M(3), s);
            %             [D_KHOSVD,X_KHOSVD] =  KHOSVD_SC_3D(Y_train,paramSC,Max_Iter_DL,D_init_k{1},D_init_k{2},D_init_k{3});
            %             %%%Dictionary training Representation Error with OMP
            %             Rep_err_train_KHOSVD(mont_crl,sigma_ind) = norm(Y_train - D_KHOSVD*X_KHOSVD,'fro')^2/numel(Y_train);
            %             %%%Dictionary test Representation Error
            %             X_test_KHOSVD = SparseCoding(Y_noisy,D_KHOSVD,paramSC);
            %             Y_KHOSVD=D_KHOSVD*X_test_KHOSVD;
            %             %%%Reconstructing the image from the overlapping patches
            %             [Image_out_KHOSVD, cnt_KHOSVD] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_KHOSVD,N_freq);
            %
            %             Rep_err_test_KHOSVD(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_KHOSVD./cnt_KHOSVD,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            %             %patch_err_test_KHOSVD(mont_crl,n_cnt) = norm(Y_clean - D_KHOSVD*X_test_KHOSVD,'fro')/N_blocks;
            %
             %% LS Updates
%             display('Training structured dictionary using LS')
%             [D_LS,X_train_LS] = LS_SC_3D(Y_train,paramSC,Max_Iter_DL,D_init_k{1},D_init_k{2},D_init_k{3});
%             %%Dictionary Representation training Error with OMP
%             Rep_err_train_LS(mont_crl,sigma_ind) = norm(Y_train - D_LS*X_train_LS,'fro')^2/numel(Y_train);
%             %%%Dictionary test Representation Error
%             X_test_LS = SparseCoding(Y_noisy,D_LS,paramSC);
%             Y_LS=D_LS*X_test_LS;
%             %%%Reconstructing the image from the overlapping patches
%             [Image_out_LS, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_LS,N_freq);
%             
%             Rep_err_test_LS(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_LS./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             %            %with OMP
%             X_test_LS_OMP = OMP(D_LS,Y_noisy,s);
%             Y_LS_OMP = D_LS*X_test_LS_OMP;
%             [Image_out_LS_OMP, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_LS_OMP,N_freq);
%             Rep_err_test_LS_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_LS_OMP./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            
            %% SeDiL
%             display('Training structured dictionary using SeDiL')
%             paramSeDiL.D{1}     = D_init_k{1};
%             paramSeDiL.D{2}     = D_init_k{2};
%             paramSeDiL.D{3}     = D_init_k{3};
%             Learn_para_SeDiL = learn_separable_dictionary(Y_tns, paramSeDiL);
%             D1_sedil = Learn_para_SeDiL.D{1};
%             D2_sedil = Learn_para_SeDiL.D{2};
%             D3_sedil = Learn_para_SeDiL.D{3};
%             D_SEDIL = kron(D3_sedil,kron(D2_sedil,D1_sedil));
%             X_train_SEDIL = reshape(Learn_para_SeDiL.X,p,N);
%             %%%Dictionary Representation training Error
%             Rep_err_train_SEDIL(mont_crl,sigma_ind) = norm(Y_train - D_SEDIL*X_train_SEDIL,'fro')^2/numel(Y_train);
%             %%%Dictionary test Representation Error
%             
%             X_test_SEDIL = SparseCoding(Y_noisy,D_SEDIL,paramSC);
%             Y_SEDIL_OMP = D_SEDIL*X_test_SEDIL;
%             %%%Reconstructing the image from the overlapping patches
%             [Image_out_SEDIL_OMP, cnt_SEDIL] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_SEDIL_OMP,N_freq);
%             Rep_err_test_SEDIL_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_SEDIL_OMP./cnt_SEDIL,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
%             Y_SEDIL = D_SEDIL*X_train_SEDIL;
%             %%%Reconstructing the image from the overlapping patches
%             [Image_out_SEDIL, cnt_SEDIL] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_SEDIL,N_freq);
%             Rep_err_test_SEDIL(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_SEDIL./cnt_SEDIL,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
            %% STARK
            disp('Training structured dictionary using STARK')
            %             lambdaADMM=[norm(Y_train,'fro')^(1.5)/10,2*norm(Y_train)^2]; %norm(Y,'fro')^(1.5)/9 9*norm(Y,'fro')^(0.5) norm(Y)^2/7]
            lambdaADMM=norm(Y_train,'fro')^(1.5)/10;
            %             gammaADMM =[lambdaADMM/100,lambdaADMM/20,lambdaADMM/10,lambdaADMM*100];% [lambda/5 lambda] %Lagrangian parameters
            gammaADMM = lambdaADMM/20;
            ParamSTARK.lambdaADMM=lambdaADMM;
            ParamSTARK.gammaADMM=gammaADMM;
            [D_STARK, X_STARK, Reconst_error_STARK] = STARK(Y_train, Permutation_vectors, D_init, ParamSTARK, paramSC);
            %Dictionary training Representation Error
            Rep_err_train_STARK(mont_crl,sigma_ind)=norm(Y_train-D_STARK*X_STARK,'fro')^2/numel(Y_train);
            %%%Dictionary test Representation Error
            X_test_STARK = SparseCoding(Y_noisy,D_STARK,paramSC);
            Y_STARK=D_STARK*X_test_STARK;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_STARK, cnt_STARK] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_STARK,N_freq);
            
            Rep_err_test_STARK(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_STARK./cnt_STARK,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            
            %             Rep_err_test_STARK(lambda,gamma)=norm(reshape(input_data-Image_out_STARK./cnt_STARK,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            %with OMP
            X_test_STARK_OMP = OMP(D_STARK,Y_noisy,s);
            Y_STARK_OMP = D_STARK*X_test_STARK_OMP;
            [Image_out_STARK_OMP, cnt_STARK] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_STARK_OMP,N_freq);
            Rep_err_test_STARK_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_STARK_OMP./cnt_STARK,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
            %% TeFDiL
            disp('Training structured dictionary using TeFDiL')
            [D_TeFDiL, X_TeFDiL,Reconst_error_TeFDiL] = TeFDiL(Y_train,Permutation_vectors, D_init, ParamTeFDiL, paramSC);
            %Dictionary training Representation Error
            Rep_err_train_TeFDiL(mont_crl,sigma_ind)=norm(Y_train-D_TeFDiL*X_TeFDiL,'fro')^2/numel(Y_train);
            %%%Dictionary test Representation Error
            X_test_TeFDiL = SparseCoding(Y_noisy,D_TeFDiL,paramSC);
            Y_TeFDiL=D_TeFDiL*X_test_TeFDiL;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL, cnt_TeFDiL] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL,N_freq);
            
            Rep_err_test_TeFDiL(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL./cnt_TeFDiL,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            %             %patch_err_test_TeFDiL(mont_crl,n_cnt,l_cnt) = norm(Y_clean - D_TeFDiL*X_test_TeFDiL,'fro')/N_blocks;
            %with OMP
            X_test_TeFDiL_OMP = OMP(D_TeFDiL,Y_noisy,s);
            Y_TeFDiL_OMP = D_TeFDiL*X_test_TeFDiL_OMP;
            [Image_out_TeFDiL_OMP, cnt_TeFDiL] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL_OMP,N_freq);
            Rep_err_test_TeFDiL_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL_OMP./cnt_TeFDiL,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
            %% Initialization for r=2
%             D_init_k={2,3};
%             for k=1:3
%                 D_initk = unfold(Y_tns,size(Y_tns),k);
%                 for r=1:2
%                     cols_k = randperm(N*m/M(k),P(k));
%                     D_init_k{r,k} = normcols(D_initk(:,cols_k));
%                 end
%             end
%             D_init2 = normc( kron(D_init_k{1,3},kron(D_init_k{1,2},D_init_k{1,1}))...
%                 +  kron(D_init_k{2,3},kron(D_init_k{2,2},D_init_k{2,1})));
%             D_init2 = normc(D_init2);
            
            %%  Sum of LS Updates
%             display('Training structured dictionary using sum of LS')
%             [D_sum_LS,X_train_sum_LS] = LS_sum_SC_3D(Y_train,2,paramSC,Max_Iter_DL,D_init_k);
%             %%Dictionary Representation training Error with OMP
%             Rep_err_train_sum_LS(mont_crl,n_cnt) = norm(Y_train - D_sum_LS*X_train_sum_LS,'fro')^2/N;
%             %%Dictionary Representation test Error with OMP
%             X_test_sum_LS = SparseCoding(Y_noisy,D_sum_LS,paramSC);
%             Y_sum_LS=D_sum_LS*X_test_sum_LS;
%             %%%Reconstructing the image from the overlapping patches
%             [Image_out_sum_LS, cnt_sum_LS] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_sum_LS,N_freq);
%             
%             Rep_err_test_sum_LS(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_sum_LS./cnt_sum_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
%             %with OMP
%             X_test_sum_LS_OMP = OMP(D_sum_LS,Y_noisy,s);
%             Y_sum_LS_OMP = D_sum_LS*X_test_sum_LS_OMP;
%             [Image_out_sum_LS_OMP, cnt_sum_LS] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_sum_LS_OMP,N_freq);
%             Rep_err_test_sum_LS_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_sum_LS_OMP./cnt_sum_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            
            %% TefDil rank2
%             disp('Training structured dictionary using TeFDiL (rank2)')
%             ParamTeFDiL.TensorRank=2;
%             [D_TeFDiL2, X_TeFDiL2, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init2, ParamTeFDiL,paramSC);
%             %%%Dictionary test Representation Error
%             X_test_TeFDiL2 = SparseCoding(Y_noisy,D_TeFDiL2,paramSC);
%             Y_TeFDiL2=D_TeFDiL2*X_test_TeFDiL2;
%             %%%Reconstructing the image from the overlapping patches
%             [Image_out_TeFDiL2, cnt_TeFDiL2] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL2,N_freq);
%             Rep_err_test_TeFDiL2(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL2./cnt_TeFDiL2,dim1,dim2*N_freq),'fro')^2/numel(input_data);
%             
%             %with OMP
%             X_test_TeFDiL2_OMP = OMP(D_TeFDiL2,Y_noisy,s);
%             Y_TeFDiL2_OMP = D_TeFDiL2*X_test_TeFDiL2_OMP;
%             [Image_out_TeFDiL2_OMP, cnt_TeFDiL2] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL2_OMP,N_freq);
%             Rep_err_test_TeFDiL2_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL2_OMP./cnt_TeFDiL2,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            %% Initialization for r=32
            Rnk=32;
            D_init_k={Rnk,3};
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                for r=1:Rnk
                    cols_k = randperm(N*m/M(k),P(k));
                    D_init_k{r,k} = normcols(D_initk(:,cols_k));
                end
            end
            D_init32 = zeros(m,p);
            for r=1:Rnk
                D_init32 = D_init32 + kron(D_init_k{r,3},kron(D_init_k{r,2},D_init_k{r,1}));
            end
            D_init32 = normc(D_init32);
            
            %% TefDil rank32
            disp('Training structured dictionary using TeFDiL (rank32)')
            ParamTeFDiL.TensorRank=Rnk;
            [D_TeFDiL32, X_TeFDiL32, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init32, ParamTeFDiL,paramSC);
            %%%Dictionary test Representation Error
            X_test_TeFDiL32 = SparseCoding(Y_noisy,D_TeFDiL32,paramSC);
            Y_TeFDiL32=D_TeFDiL32*X_test_TeFDiL32;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL32, cnt_TeFDiL32] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL32,N_freq);
            Rep_err_test_TeFDiL32(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL32./cnt_TeFDiL32,dim1,dim2*N_freq),'fro')^2/numel(input_data);
            
            %with OMP
            X_test_TeFDiL32_OMP = OMP(D_TeFDiL32,Y_noisy,s);
            Y_TeFDiL32_OMP = D_TeFDiL32*X_test_TeFDiL32_OMP;
            [Image_out_TeFDiL32_OMP, cnt_TeFDiL32] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL32_OMP,N_freq);
            Rep_err_test_TeFDiL32_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL32_OMP./cnt_TeFDiL32,dim1,dim2*N_freq),'fro')^2/numel(input_data);
                        
            %% Saving Results
            save('house_rnd1_8MonteCarl_fliplr','sig_vals','M','P',...                                               
                'Rep_err_train_STARK','Rep_err_test_STARK',...
                'Rep_err_train_TeFDiL','Rep_err_test_TeFDiL',...
                'Rep_err_train_TeFDiL32','Rep_err_test_TeFDiL32',...                
                'Rep_err_test_STARK_OMP',...
                'Rep_err_test_TeFDiL_OMP',...
                'Rep_err_test_TeFDiL32_OMP')
        end
    end
end