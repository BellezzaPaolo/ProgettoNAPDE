function [costHistory,y]=StocasticGradientDescent(data)
costHistory=zeros(1,data.Maxiter);

n_parameters=CountParameters(data);

y=0.5*randn(n_parameters,1);

for ii=1:data.Maxiter
    [costHistory(ii),df]=FandG(data,y);
    y=y-data.eta*df;
end
end