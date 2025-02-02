function [data] = Dati(test)
%=======================================================================================================
% TEMPLATE OF THE STRUCT DATI
%=======================================================================================================
%
%  DATA= struct( 'name',              % set the name of the test
%                'x',                 % set the points of the inputs of the NN
%                'y',                 % set the points of the outputs of the NN
%                'shape',             % set the number of neurons for every layer of the NN [n1,n2,n2,...]
%                'L',                 % set the depth of the NN
%                'sigma',             % set the activation function of every neuron of the NN
%                'sigmaprime',        % set the derivativeof the activation function
%                'eta',               % set the value of the learning rate
%                'batchsize',         % set the size of the batch (the number of points used in the training)
%                'Maxiter',           % set the maximum number of iteration made by the method
%                'n_parareal',        % set the number of iteration of the parareal algorithm
%                'n_coarse',          % set the number of sub interval of my domain 
%                );
%========================================================================================================


if strcmp(test,'TestSeno')
    data = struct( 'name',             test,...
                   ... % Test name
                   'x',               linspace(0,pi,20),...
                   ...%array of the input points of the neural network
                   'y',               sin(3.*linspace(0,pi,20)),...
                   ...%array of the uoputs of the neural network
                   'shape',           [1,16,32,16,1],...
                   ...%shape of the neural network
                   'L',               4,...
                   ...%length of the neural network
                   'sigma',           @(x) tanh(x),...
                   ...% activation function of every neuron
                   'sigmaprime',      @(x) 1-tanh(x).^2,...
                   ...% derivative of the activation function
                   'eta',             0.1,...
                   ...%learning rate
                   'batchsize_gradient',20,...
                   ...%
                   'batchsize_coarse',20,...
                   ...%
                   'batchsize_fine',  20,...
                   ...%
                   'Maxiter',         8e5,...
                   ...%
                    'n_fine',         10,...
                   ...%
                   'n_coarse',        2000,...
                   ...%
                   'dT',              0.01,...
                   ...%
                   'dt',              0.01 ...
                   ...%                  
                   );
elseif strcmp(test,'TestHigham')
    data = struct( 'name',             test,...
                   ... % Test name
                   'x',               [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7; ...
                                       0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6],...
                   ...%array of the input points of the neural network
                   'y',               [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)],...
                   ...%array of the uoputs of the neural network
                   'shape',           [2,3,3,2],...
                   ...%shape of the neural network
                   'L',               4,...
                   ...%length of the neural network
                   'sigma',           @(x) 1./(1+exp(-x)),...
                   ...%activation function of every neuron
                   'sigmaprime',      @(x) 1./(1+exp(-x)).^2.*(exp(-x)),...
                   ...%derivative of the activation function
                   'eta',             0.01,...
                   ...%learning rate
                   'batchsize_gradient',10,...
                   ...%
                   'batchsize_coarse',10,...
                   ...%
                   'batchsize_fine',  10,...
                   ...%
                   'Maxiter',         1e9,...
                   ...%
                   'n_fine',          50,...
                   ...%
                   'n_coarse',        2000,...
                   ...%
                   'dT',              0.3,...
                   ...%
                   'dt',              0.3/50 ...
                   ...%
                   );
elseif strcmp(test,'TestPolinomio')
    data = struct( 'name',             test,...
                   ... % Test name
                   'x',               linspace(-1,1,20),...
                   ...%array of the input points of the neural network
                   'y',                 0.5.*linspace(-1,1,20).^3-2.*linspace(-1,1,20).^2+1,...
                   ...%array of the uoputs of the neural network
                   'shape',           [1,16,32,16,1],...
                   ...%shape of the neural network
                   'L',               3,...
                   ...%length of the neural network
                   'sigma',           @(x) 1./(1+exp(-x)),...
                   ...%activation function of every neuron
                   'sigmaprime',      @(x) 1./(1+exp(-x)).^2.*(exp(-x)),...
                   ...%derivative of the activation function
                   'eta',             0.001,...
                   ...%learning rate
                   'batchsize_gradient',20,...
                   ...%
                   'batchsize_coarse',20,...
                   ...%
                   'batchsize_fine',  20,...
                   ...%
                   'Maxiter',         2.5e5,...
                   ...%
                   'n_fine',          2000,...
                   ...%
                   'n_coarse',        2000,...
                   ...%
                   'dT',              0.001,...
                   ...%
                   'dt',              0.001 ...
                   ...%
                   );
elseif strcmp(test,'TestSmorzato')
    data = struct( 'name',             test,...
                   ... % Test name
                   'x',               linspace(0,10,30),...
                   ...%array of the input points of the neural network
                   'y',               exp(-linspace(0,10,30)).*sin(linspace(0,10,30)),...
                   ...%array of the uoputs of the neural network
                   'shape',           [1,16,16,1],...
                   ...%shape of the neural network
                   'L',               4,...
                   ...%length of the neural network
                   'sigma',           @(x) 1./(1+exp(-x)),...
                   ...%activation function of every neuron
                   'sigmaprime',      @(x) 1./(1+exp(-x)).^2.*(exp(-x)),...
                   ...%derivative of the activation function
                   'eta',             0.001,...
                   ...%learning rate
                   'batchsize_gradient',30,...
                   ...%
                   'batchsize_coarse',30,...
                   ...%
                   'batchsize_fine',  30,...
                   ...%
                   'Maxiter',         1e7,...
                   ...%
                   'n_fine',          2000,...
                   ...%
                   'n_coarse',        2000,...
                   ...%
                   'dT',              0.001,...
                   ...%
                   'dt',              0.001 ...
                   ...%
                   );
end
end
