function [f,y1]=coarse_solver(t0,y0,dT,data)
%=======================================================================================================
% solves the training on a coarse grid using Forward Euler
%=======================================================================================================
% INPUTS:
%   -t0: starting time
%   -y0: N-dimensional vector that rappresents the point where calculate
%        the value of the function and its gradient and contains weights and
%        bias of the NN
%   -dT: time pass of the discretization
%   -data: struct that cointans every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -f: value of the Loss function in the given point y0
%   -y1: vector that contains the weights and biases at the end of the
%       training
%=======================================================================================================
    index=randperm(size(data.x,2),data.batchsize_coarse);    
    global iterCoarse
    iterCoarse=iterCoarse+1;

    [f,df] = FandG(data,y0,index);
    y1=y0-dT*df;
end