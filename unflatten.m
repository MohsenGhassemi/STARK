function Y = unflatten( X, dim, sizex )
% UNFLATTEN( X, DIM, SIZE )   Un-Flattens the matrix X 
%
% Unflattens in the dimension specified, which results in
% a tensor of SIZE. Flattening and Un-Flattening are used
% in the multilinear svd.
%
% Author: Greg Coombe
% Date: July 14, 2003
%

%if ( prod( size(X)) ~= prod(sizex) )
%    error('Matrix sizes do not match.');
%end

n1 = sizex(1);
n2 = sizex(2);
n3 = sizex(3);

% Loop over the dimension, copying other dimensions in place
if ( dim == 1 )
    for i = 1:n2
      Y(:,i,:) = X(:, ((i-1)*n3 + 1):(i*n3));
    end
elseif ( dim == 2 )
    for i = 1:n3
      Y(:,:,i) = X(:, ((i-1)*n1 + 1):(i*n1))';
    end
elseif ( dim == 3 )
    for i = 1:n1
      Y(i,:,:) = X(:, ((i-1)*n2 + 1):(i*n2))';
    end
else
    error('Function not defined for dimensions > 3.');
end
  
        