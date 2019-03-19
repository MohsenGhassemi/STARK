
%%% Mohsen Ghassemi
%%% The demo of the low-separation-rank algorithm STARK on the "house" image.

% Y: observation matrix
% D: dictionary
% X: coefficient matrix
% lambda: regularization term
% K: tensor order
% gamma: Augmented Lagrangian multiplier
% Objective function
%F_D = (1/2)*norm(Y-D*X,'fro')^2 + (lambda/N)*sum([unfold(W_1,1),unfold(W_2,2),unfold(W_3,3)]);
clc; clear; close all

% Install the FISTA Toolbox in the same directory as this file
addpath(genpath('FISTA-SPAMS'));


%% Loading image and extracting overlapping patches from it
Image=double(imread('house_color.tiff'));

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


step=2; %Stride
for i=0:step:dim1-patch_size% step determines how much of the neighboring patches overlap
    for j=0:step:dim2-patch_size
        block_data{i/step+1,j/step+1}=input_data(i+1:i+patch_size, j+1:j+patch_size,:);
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
        obsrvtn_vect(:,k) = reshape(obsrvtn_tensor(:,:,:,k),patch_size^2*N_freq,1,1);%each point is vectorized
    end
end

%%% Noisy and noiseless data
Y_clean = obsrvtn_vect; %clean data

%% Dictionary Parameters
M = [patch_size, patch_size, N_freq];
P =[2*patch_size ,2*patch_size, N_freq];

Dictionary_sizes{1}=fliplr(M);
Dictionary_sizes{2}=fliplr(P);

m=prod(M);
p=prod(P);

%% Experiment setup
sample_size=length(Y_clean);
length_sample_sizes=length(sample_size);



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



% STARK Parameters
ParamSTARK.TolADMM = 1e-4; %tolerance in ADMM update
ParamSTARK.MaxIterADMM = 10;
ParamSTARK.DicSizes=Dictionary_sizes;
ParamSTARK.Sparsity=s;
ParamSTARK.MaxIterDL=Max_Iter_DL;
ParamSTARK.TolDL=10^(-4);



sigma = 50;
Y_noisy=Y_clean+sigma/max(max(max(Image)))*randn(size(Y_clean)); % noisy data


%Generate training data
Y_train = Y_noisy(:,randperm(N_blocks,sample_size));

%initializing subdictionaries
D_init_k={1,3};

Y_tns = reshape(Y_train,M(1),M(2),M(3),N);
for k=1:3
    D_initk = unfold(Y_tns,size(Y_tns),k);
    cols_k = randperm(N*prod(M)/M(k),P(k));
    D_init_k{k} = normcols(D_initk(:,cols_k));
end
D_init = kron(D_init_k{3},kron(D_init_k{2},D_init_k{1}));


%% STARK
disp('Training structured dictionary using STARK')
lambdaADMM=norm(Y_train,'fro')^(1.5)/10;
gammaADMM = lambdaADMM/20;
ParamSTARK.lambdaADMM=lambdaADMM;
ParamSTARK.gammaADMM=gammaADMM;
[D_STARK, X_STARK, Reconst_error_STARK] = STARK(Y_train, Permutation_vectors, D_init, ParamSTARK, paramSC);

%Dictionary training Representation Error
Rep_err_train_STARK=norm(Y_train-D_STARK*X_STARK,'fro')^2/numel(Y_train);

%%%Dictionary test Representation Error
X_test_STARK = SparseCoding(Y_noisy,D_STARK,paramSC);
Y_STARK=D_STARK*X_test_STARK;

%%%Reconstructing the image from the overlapping patches
[Image_out_STARK, cnt_STARK] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_STARK,N_freq);
Rep_err_test_STARK=norm(reshape(input_data-Image_out_STARK./cnt_STARK,dim1,dim2*N_freq),'fro')^2/numel(input_data);

%with OMP
X_test_STARK_OMP = OMP(D_STARK,Y_noisy,s);
Y_STARK_OMP = D_STARK*X_test_STARK_OMP;
[Image_out_STARK_OMP, cnt_STARK] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_STARK_OMP,N_freq);
Rep_err_test_STARK_OMP=norm(reshape(input_data-Image_out_STARK_OMP./cnt_STARK,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            
