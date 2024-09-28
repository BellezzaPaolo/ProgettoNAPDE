function [cost,y0] = fine_solver(t0, y0, dt, n_fine,data)
%=======================================================================================================
% solves the training on a finer grid using Forward Euler
%=======================================================================================================
% INPUTS:
%   -t0: starting time
%   -y0: N-dimensional vector that rappresents the point where calculate
%        the value of the function and its gradient and contains weights and
%        bias of the NN
%   -dT: time pass of the discretization
%   -n_fine: number of refinements
%   -data: struct that cointans every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -f: value of the Loss function in the given point y0
%   -y1: vector that contains the weights and biases at the end of the
%       training
%=======================================================================================================
cost=zeros(n_fine,1);
for ii=1:n_fine
    index=randperm(size(data.x,2),data.batchsize);
    [f,df]=FandG(data,y0,index);
    y0=y0-dt*df;
    cost(ii)=f;
end
end