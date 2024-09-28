function [costHistory,y]=StocasticGradientDescent(data)
%=======================================================================================================
% Applies the gradient descent algorithm to the neural network
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -costHistory: vector containing the values of the Loss function
%   -y: vector cointaining weights and biases of the neural network
%=======================================================================================================
costHistory=zeros(1,data.Maxiter);

% count the number of parameters in the NN
n_parameters=CountParameters(data);

y=0.5*randn(n_parameters,1);

for ii=1:data.Maxiter
    index=randperm(size(data.x,2),1);
    [costHistory(ii),df]=FandG(data,y,index);
    y=y-data.eta*df;
end
end