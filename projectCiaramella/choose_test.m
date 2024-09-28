function [L,xx,yy,nn,J,sig,d_sig,alpha,dim,x,yd] = choose_test(example)

if strcmp(example,'sin')
    L = 4;                             % total number of layers
    nn = [1,3,3,1];                    % structure of the neural network
    J=1;
    xx=linspace(0,pi,J);
    yy = sin(3*xx);
    sig = @(x) tanh(x);                % activation function 
    d_sig = @(x) 1 - tanh(x).^2;       % derivative of sigmoid function 
    alpha = 0.0;                       % regularization parameter
end


if strcmp(example,'logsin')
    L = 4;                             % total number of layers
    nn = [1,3,3,1];                    % structure of the neural network
    J=20;
    xx=linspace(0,pi,J);
    yy = log10(xx+1).*sin(3*xx);
    sig = @(x) tanh(x);                % activation function 
    d_sig = @(x) 1 - tanh(x).^2;       % derivative of sigmoid function 
    alpha = 0.0;                       % regularization parameter
end

if strcmp(example,'Higham')
    L = 4;                             % number of layers
    nn = [2,3,3,2];                    % structure of the neural network
    % J=11;
    % x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7,0.3]; 
    % x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6,0.7];
    % xx=[x1; x2];
    % yy = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];
    J=10;
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7,0.3]; 
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6,0.7];
    xx=[x1; x2];
    yy = [ones(1,5) zeros(1,6); zeros(1,5) ones(1,6)];
    sig = @(x) 1./(1+exp(-x));         % activation function 
    d_sig = @(x) sig(x).*(1-sig(x));   % derivative of sigmoid function 
    alpha = 0.0;                       % regularization parameter
end

dim = sum(nn(1:end-1).*nn(2:end))+sum(nn(2:end)); % total dimension of the problem
for j=1:J                              % set-up input/output data
    x{j}=xx(:,j);
    yd{j}=yy(:,j);
end
dim

end