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

rng(50)
y=0.5*randn(n_parameters,1);

convergence=true;
global iterGrad
ii=1;
while convergence
    tic
    index=randperm(size(data.x,2),data.batchsize_gradient);
    [f,df]=FandG(data,y,index);
    costHistory(ii)=costo(data,y);
    y1=y-data.eta*df;

    iterGrad=iterGrad+1;
    convergence=f>7*10^-3 && ii<data.Maxiter; %norm(y1-y,2)>10^-5; %full stocastic f>10^-5  % altro stopping criteria df>10^-5
    ii=ii+1;
    time_iter = toc;
    if stampa && mod(ii,10000)==0
        disp(['iteration ' num2str(ii) ', time: ', num2str(time_iter) ', cost_history = ', num2str(costHistory(ii-1)),' norm ',num2str(norm(y1-y,2))])
    end
    y = y1;
end
costHistory=costHistory(1:ii-1);
end