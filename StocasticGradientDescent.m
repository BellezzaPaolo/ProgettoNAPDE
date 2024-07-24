function [costHistory,y]=StocasticGradientDescent(data)
costHistory=zeros(1,data.Maxiter);

n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end

y=0.5*randn(n_parameters,1);

for ii=1:data.Maxiter
    [costHistory(ii),df]=FandG(data,y);
    y=y-data.eta*df;
end
end