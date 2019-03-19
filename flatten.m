function Y = flatten( X, dim )
% FLATTEN( X, DIM )   Tensor Flatten
%
% Flattens the tensor X in the dimension specified.
% Flattening is used for multilinear tensor problems like svd.
%
% Author: Greg Coombe
% Date: July 14, 2003
%

if ( dim > ndims(X) )
    error('Index outside dimensions.');
elseif ( ndims(X) > 3 )
    error('Function not defined for dimensions > 3.');
end

% Loop over the dimension, copying other dimensions in place
    
if ( dim == 1 )
    Y = squeeze(X(:,1,:));
    for i = 2:size(X,2),
      Y = [Y squeeze(X(:,i,:)) ];
    end
elseif ( dim == 2 )
    Y = X(:,:,1)';
    for i = 2:size(X,3),
      Y = [Y X(:,:,i)' ];
    end
elseif ( dim == 3 )
    Y = shiftdim(X(1,:,:),1)';
    for i = 2:size(X,1),
      Y = [Y shiftdim(X(i,:,:),1)' ];
    end
end
  
        