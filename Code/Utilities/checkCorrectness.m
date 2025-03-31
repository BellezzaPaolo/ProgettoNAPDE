function [] = checkCorrectness(data)
%=======================================================================================================
% Apply finite centrate difference in order to test the output of the
% function FandG ( df(theta) = ( f(theta+h) - f(theta-h) ) / (2*h) )
%=======================================================================================================
% INPUTS:
%   -data: (struct) contains every parameters of the problem (see Dati.m)
%=======================================================================================================
% Count the number of parameters of the network
n_parameters=CountParameters(data);

% Random starting point for the test
%rng('default')

% Initialize the points and the discretizations step
theta=rand(n_parameters,1);
h=rand(n_parameters,1);
eps=1;
index=randperm(size(data.x,2),data.batchsize_gradient);

err=zeros(1,4);

% Compute the derivative df(theta)
[~,df]=FandG(data,theta,index);

% Iterate to check the second order of the error in the approximation 
for ii=1:6
    % Compute f(theta+h)
    [fp,~]=FandG(data,theta+eps*h,index);
    % Compute f(theta-h)
    [fm,~]=FandG(data,theta-eps*h,index);
    
    % Apply the formula
    err(ii)=abs((fp-fm)/(2.*eps)-df'*h);

    fprintf('eps %12d error   %d \n',eps,err(ii));

    eps=eps*0.1;
end
end