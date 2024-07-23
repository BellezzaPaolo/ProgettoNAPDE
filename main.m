clear
close all
clc

data= Dati('Test2');

n=0;
for l=1:data.L-1
    n=n+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end

y=randn(n,1);
for ii=1:data.Maxiter
    [f(ii),df]=FandG(data,y);
    y=y+data.eta*df;
end


semilogy(1:1e3:data.Maxiter,f(1:1e3:end))
checkCorrectness(data);
