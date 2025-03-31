function [cost,costExact,y0] = fine_solver(y0, dt, n_fine,data,stampa)
%=======================================================================================================
% Makes n_fine steps of Forward Euler on the fine grid
%=======================================================================================================
% INPUTS:
%   -y0:        (vector of double) weights and biases of the neural network
%   -dt:        (double) time pass of the discretization also called learning rate
%   -n_fine:    (int) number of step to compute
%   -data:      (struct) contains every parameters of the problem (see Dati.m)
%   -stampa:    (bool) decides if the cost function must be computed exactly
%               or not
% OUTPUTS:
%   -f:         (double) value of the cost function computed in the points
%               of the batch used in this update
%   -CostExact: (double) value of the cost function computed on all the
%               training set (only for testing porpouses)
%   -y1:        (vector of double) updated weights and biases
%=======================================================================================================
cost=zeros(n_fine,1);
costExact = zeros(n_fine,1);

global iterFine

% Loop over the n_fine steps
for ii=1:n_fine
    % Extract the points for this update
    index=randperm(size(data.x,2),data.batchsize_fine);
    % Compute cost function and its gradient
    [cost(ii),df]=FandG(data,y0,index);
    % Compute the exact value of the cost function
    if stampa
        costExact(ii)= costo(data,y0);
    else
        costExact(ii) = cost(ii);
    end
    % Update the global counter for testing
    iterFine=iterFine+1;
    % Update the network parameters
    y0=y0-dt*df;
end
end