function [f,CostExact,y1]=coarse_solver(y0,dT,data,stampa)
%=======================================================================================================
% Makes one step of Forward Euler on the coarse grid
%=======================================================================================================
% INPUTS:
%   -y0:        (N-dimensional vector of double) point where calculate
%               the value of the cost function and its gradient and contains weights and
%               bias of the neural network
%   -dT:        (double) time pass of the discretization also called learning rate
%   -data:      (struct) contains every parameters of the problem (see Dati.m)
%   -stampa:    (bool) decides if the cost function must be computed exactly
%               or not
% OUTPUTS:
%   -f:         (double) value of the cost function computed in the points
%               of the batch used in this update
%   -CostExact: (double) value of the cost function computed on all the
%               training set (only for testing porpouses)
%   -y1:        (vector of double) that contains the updated weights and biases
%=======================================================================================================
    % Choose randomly the points of the batch to use
    index=randperm(size(data.x,2),data.batchsize_coarse);

    % Update the global counter for testing
    global iterCoarse
    iterCoarse=iterCoarse+1;
    
    % Compute the value and the gradient of the cost function
    [f,df] = FandG(data,y0,index);
    % Compute the exact value of the cost function
    if stampa
        CostExact = costo(data,y0);
    else
        CostExact = f;
    end
    % Update the network parameters
    y1=y0-dT*df;
end