clear all; close all;clc;
format long;

% choose example
example = 'Higham';
%example = 'sin';
%example = 'logsin';
[L,xx,yy,nn,J,sig,d_sig,alpha,dim,x,yd] = choose_test(example);


% collect data - Problem
data.x = x;
data.yd = yd;
data.sig = sig;
data.d_sig = d_sig;
data.alpha = alpha;
data.L = L;
data.nn = nn;
data.J = J; 



% test gradient vectorized structure
theta = 1-2*rand(dim,1);
dtheta = 1-2*rand(dim,1);
setJ=1:J;
data.setJ=setJ;
[df,f] = FandG(theta,data);
%[f2,df2] = FandG_2(data,theta);
right = dtheta'*df;
epsi=1;
for j=1:5
    thetaP=theta+epsi*dtheta;
    thetaM=theta-epsi*dtheta;
    [~,fP] = FandG(thetaP,data);
    [~,fM] = FandG(thetaM,data);
    left = (fP-fM)/(2*epsi);
    test = abs(left-right);
    fprintf( '%e   %e\n', epsi,test)
    epsi=epsi*0.1;
end 

