function [f,df]=FandG(data,y0,index)
%=======================================================================================================
% Computes the value of the cost function f and its gradient in the given point y0
%=======================================================================================================
% INPUTS:
%   -data:      (struct) contains every parameters of the problem (see Dati.m)
%   -y0:        (vector of double) weights and biases of the neural network
%   -index:     (vector of int) indexes of the points in the batch used in this iteration of the training
% OUTPUTS:
%   -f:         value of the cost function in the given point y0
%   -df:        N-dimensional vector containing the gradient of the cost
%               function valueted in the point y0
%=======================================================================================================

% Initialize everything
L=data.L;
x=data.x;
y=data.y;
sigma=data.sigma;
sigmaprime=data.sigmaprime;

W=cell(L,1);
b=cell(L,1);
a=cell(L,1);
delta=cell(L,1);
z=cell(L,1);

% Compose the set of matrices and weights from the vector y0
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

f=0;
df=zeros(size(y0));

for j=1:size(index,2)
    % Forward pass of the network
    a{1}=x(:,index(j));%j);
    for l=2:L
        z{l}=W{l}*a{l-1}+b{l};
        a{l}=sigma(z{l});
    end
    
    % Compute the cost function
    f=f+0.5*norm(a{end}-y(:,index(j))).^2;
    
    % Back propagation of the network using chain rule
    delta{L}=sigmaprime(z{L}).*(a{end}-y(:,index(j)));
    for l=L-1:-1:2
        delta{l}=sigmaprime(z{l}).*(W{l+1}'*delta{l+1});
    end

    % Reorder the gradient in the vector df 
    pointer_df=1;
    for l=2:L
        df(pointer_df:pointer_df+data.shape(l-1)*data.shape(l)-1)=df(pointer_df:pointer_df+data.shape(l-1)*data.shape(l)-1)+reshape(delta{l}*a{l-1}',[],1);
        pointer_df=pointer_df+data.shape(l-1)*data.shape(l);
    end
    for l=2:L
        df(pointer_df:pointer_df+data.shape(l)-1)=df(pointer_df:pointer_df+data.shape(l)-1)+delta{l};
        pointer_df=pointer_df+data.shape(l);
    end
end