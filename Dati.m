function [data] = Dati(test)

if test=='Test1'
    data = struct( 'name',             test,...
                   ... % Test name
                   'input_point',     linspace(0,pi,20),...
                   ...%array of the input points of the neural network
                   'output_point',    sin(3.*linspace(0,pi,20)),...
                   ...%array of the uoputs of the neural network
                   'shape',           [1,3,3,1],...
                   ...%shape of the neural network
                   'L',               4,...
                   ...%length of the neural network
                   'sigma',           @(x) tanh(x),...
                   ...%activation function of every neuron
                   'sigmaprime',      @(x) 1-tanh(x).^2,...
                   ...%
                   'eta',             0.75,...
                   ...%
                   'Maxiter',         1e5,...
                   ...%
                   'index',           randperm(20,20)...
                   ...%
                   );
elseif test=='Test2'
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
                   'sigmaprime',      @(x) 1./(1+exp(-x)).*(1-1./(1+exp(-x))),...
                   ...%
                   'eta',             0.05,...
                   ...%
                   'Maxiter',         1e4,...
                   ...%
                   'batchsize',           [10]...
                   ...%
                   );
end
end