function [n_parameters]= CountParameters(data)
%=======================================================================================================
% Count the number of parameters in the neural network
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -n_parameters:  (int) number of parameters
%=======================================================================================================
n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end
end