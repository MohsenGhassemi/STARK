function [Image_out, cnt] = ImageRecon(input_data,N_blocks,dim2_block,step,patch_size,Y_recon,N_freq)

Image_out=zeros(size(input_data));
cnt=zeros(size(input_data));

for k=1:N_blocks
    r_block=ceil(k/dim2_block);%index on the block
    c_block=k-floor((k-1)/dim2_block)*dim2_block;
    
    r_image=(r_block-1)*step;% index on the image
    c_image=(c_block-1)*step;
    
    Image_out(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)=...
        Image_out(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)+...
        reshape(Y_recon(:,k),patch_size,patch_size,N_freq);
    cnt(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)=cnt(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)+1;
end

%         Image_KSVD=zeros(size(input_data));
%         cnt=zeros(size(input_data));
%
%         for k=1:N_blocks
%             r_block=ceil(k/dim2_block);%index on the block
%             c_block=k-floor((k-1)/dim2_block)*dim2_block;
%
%             r_image=(r_block-1)*step;% index on the image
%             c_image=(c_block-1)*step;
%
%             Image_KSVD(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)=...
%                 Image_KSVD(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)+...
%                 reshape(Y_KSVD(:,k),patch_size,patch_size,N_freq);
%             cnt(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)=cnt(r_image+1:r_image+patch_size,c_image+1:c_image+patch_size,:)+1;
%
%         end