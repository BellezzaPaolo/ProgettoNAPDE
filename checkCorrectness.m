function [] = checkCorrectness(data)
%=======================================================================================================
% Check the correctness of the function FandG
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
%=======================================================================================================
n_parameters=CountParameters(data);

% random starting point for the test
%rng('default')
theta=rand(n_parameters,1);
h=rand(n_parameters,1);
eps=1;
index=randperm(size(data.x,2),data.batchsize_gradient);

err=zeros(1,4);
[~,df]=FandG(data,theta,index);

for ii=1:6
    [fp,~]=FandG(data,theta+eps*h,index);
    [fm,~]=FandG(data,theta-eps*h,index);
    
    err(ii)=abs((fp-fm)/(2.*eps)-df'*h);

    fprintf('eps %12d error   %d \n',eps,err(ii));

    eps=eps*0.1;
end
end