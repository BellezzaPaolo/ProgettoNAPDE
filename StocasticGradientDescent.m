function [costHistory,y]=StocasticGradientDescent(data,stampa)
%=======================================================================================================
% Applies the gradient descent algorithm to the neural network
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -costHistory: vector containing the values of the Loss function
%   -y: vector cointaining weights and biases of the neural network
%=======================================================================================================
costHistory=ones(1,data.Maxiter);

% count the number of parameters in the NN
n_parameters=CountParameters(data);

y=0.5*randn(n_parameters,1);

convergence=true;

ii=1;
while convergence
    tic
    index=randperm(size(data.x,2),data.batchsize_gradient);
    [costHistory(ii),df]=FandG(data,y,index);
    y=y-data.eta*df;
    convergence=costHistory(ii)>10^-3 && ii<data.Maxiter;
    ii=ii+1;
    time_iter = toc;
    if stampa && mod(ii,10000)==0
        disp(['iteration ' num2str(ii) ', time: ', num2str(time_iter) ', cost_history = ', num2str(costHistory(ii-1))])
    end
end
costHistory=costHistory(1:ii-1);
end