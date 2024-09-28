function [f,df]=FandG2(data,y0)
%=======================================================================================================
% Computes the value of the Loss function f and the gradient in the given point y0
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (se Dati.m)
%   -y0: N-dimensional vector that rappresents the point where calculate
%        the value of the function and its gradient and contains weights and
%        bias of the NN
% OUTPUTS:
%   -f: value of the Loss function in the given point y0
%   -df: N-dimensional vector containing the derivative of the Loss
%        function valueted int he point y0
%=======================================================================================================

L=data.L;
index=10;%randperm(size(x,2),data.batchsize);
x=data.x;
y=data.y;
sigma=data.sigma;
sigmaprime=data.sigmaprime;

W=cell(L,1);
b=cell(L,1);
a=cell(L,1);
delta=cell(L,1);
z=cell(L,1);

% compose the set of matrices and weights from the vector y0
pointer_y0=0;
% assign weights matrices
for ii=2:L
    W{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii)),data.shape(ii),data.shape(ii-1));
    pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
end
% assign bias vector
for ii=2:L
    b{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii)),data.shape(ii),1);
    pointer_y0=pointer_y0+data.shape(ii);
end

f=0;
df=zeros(size(y0));


for j=1:index
    % forward pass
    a{1}=x(:,j);
    for l=2:L
        z{l}=W{l}*a{l-1}+b{l};
        a{l}=sigma(z{l});
        % D{l}=diag(sigmaprime(z{l}));
    end
    
    % compute the cost function
    f=f+0.5*norm(a{end}-y(:,j)).^2;
    
    % compute the delta and back propagation
    delta{L}=sigmaprime(z{L}).*(a{end}-y(:,j));
    for l=L-1:-1:2
        delta{l}=sigmaprime(z{l}).*(W{l+1}'*delta{l+1});
    end

    %reorder the gradient in the vector df 
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
end