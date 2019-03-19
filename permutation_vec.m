function [permutation_vector, permutation_vectorT]=permutation_vec(Dictionary_sizes)

M=Dictionary_sizes{1};
P=Dictionary_sizes{2};

if length(M)==2
    M=[1 M];
    P=[1 P];
end

m = prod(M);
p = prod(P);

permutation_vector=zeros(m*p,1);
permutation_vectorT=zeros(m*p,1);

size_D=[m, p];
size_dpi = fliplr(M.*P);

for l=1:m*p

[i,j]=ind2sub(size_D,l);


S_i_q = floor((i-1)/(M(2)*M(3)));
S_j_q = floor((j-1)/(P(2)*P(3)));

i_q = i - S_i_q*M(2)*M(3);
j_q = j - S_j_q*P(2)*P(3);

S_i_r = floor((i_q - 1)/M(3));
S_j_r = floor((j_q - 1)/P(3));


i_r = i_q - S_i_r*M(3);
j_r = j_q - S_j_r*P(3);

s = S_i_q + S_j_q*M(1) + 1;%slice
r = S_i_r + S_j_r*M(2) +1;%column
q = i_r + (j_r - 1)*M(3);%row


lp=sub2ind(size_dpi,q,r,s);

permutation_vector(l) = lp;%for lp ---> l (Dp ---> D )
permutation_vectorT(lp) = l;%for l ---> lp (D ---> Dp )
end