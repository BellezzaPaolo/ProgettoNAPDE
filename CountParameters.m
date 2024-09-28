function [n_parameters]= CountParameters(data)
%=======================================================================================================
% count the number of parameters in our NN
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
% OUTPUTS:
%   -n_parameters: double that is the number of parameters
%=======================================================================================================
n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end
end