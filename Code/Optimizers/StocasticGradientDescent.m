function [costHistory,y]=StocasticGradientDescent(data,stampa)
%=======================================================================================================
% Apply the Gradient Descent (GD) or Stochastic Gradient Descent (SGD) depending on the batch size 
% at the given neural network
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -stampa:        (bool) decides if compute exactly the value of the cost function and show in the command 
%                   window the training behaviour or not
% OUTPUTS:
%   -costHistory:   (vector of double) values of the cost function
%   -y:             (vector of double) weights and biases at the end of the training
%=======================================================================================================
costHistory=ones(1,data.Maxiter);

% Count the number of parameters in the network
n_parameters=CountParameters(data);

% Fix random seed for reproducibility
rng(50)
y=(1-2*rand(n_parameters,1));

convergence = true;
% Initiaalize global counter
global iterGrad
iterGrad = 0;
ii=1;
% Loop untill convergence
while convergence
    tic
    % Extract the points used in this training epoch
    index=randperm(size(data.x,2),data.batchsize_gradient);

    % Compute the value of the cost function and its gradient
    [costHistory(ii),df]=FandG(data,y,index);
    
    % Compute exactly the cost function
    if stampa
        costHistory(ii)=costo(data,y);
    end
    % Update weight and biases
    y1=y-data.eta*df;
    
    % Update the global counter for testing porpouses
    iterGrad=iterGrad+1;

    % Check convergence
    convergence = costo(data,y1) > data.threshold && iterGrad*data.batchsize_gradient < data.Maxiter;
    %convergence = norm(y1-y,2) > 2.5*10^-9; % Other possible stopping citeria based on the difference between to successive iterate

    ii=ii+1;

    % Print in the command window (optional)
    time_iter = toc;
    if (stampa && mod(ii,10000)==0)
        disp(['iteration ' num2str(ii) ', time: ', num2str(time_iter) ', cost_history = ', num2str(costHistory(ii-1)),', norm = ',num2str(norm(y1-y,2))])
    end

    y = y1;
end

costHistory=costHistory(1:ii-1);
end