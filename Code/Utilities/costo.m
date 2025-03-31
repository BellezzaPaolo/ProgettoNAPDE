function [f] = costo(data,y0)
%=======================================================================================================
% Computes the exact value of the Loss function i.e. a forward pass of the
% neural network of the whole training set
%=======================================================================================================
% INPUTS:
%   -y0:        (vector of double) weights and biases of the neural network
%   -data:      (struct) contains every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -f:         (double) value of the cost function computed on all the
%               training set
%=======================================================================================================
L = data.L;
sigma=data.sigma;
pointer_y0=0;

% Assign weights matrices
for ii=2:L
    W{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii)),data.shape(ii),data.shape(ii-1));
    pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
end
% Assign bias vector
for ii=2:L
    b{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii)),data.shape(ii),1);
    pointer_y0=pointer_y0+data.shape(ii);
end

% Forward step of the neural network for every point in the training set
f = 0;
for ii = 1:size(data.x,2)
    a{1}=data.x(:,ii);
    for l=2:L
        z{l}=W{l}*a{l-1}+b{l};
        a{l}=sigma(z{l});
    end

    % Compute the cost function
    f=f+0.5*norm(a{end}-data.y(:,ii)).^2;
end