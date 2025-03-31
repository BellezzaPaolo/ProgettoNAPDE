function [costi_full,med_nCoarse_full,M_full]= TestRandomness(data,learning_rate,n_fine,batch_size)
%=======================================================================================================
% Test the efficiency of the ParaFlowS algorithm respect to SGD or GD varing the learning rate 
% (data.eta and data.dt), number of refinements associated to the fine operator (data.n_fine) 
% and the batch size (data.batchsize_gradient and data.batchsize_fine)
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -learning_rate: (vector of double) learning rate to test
%   -n_fine:        (vector of int) n_fine to test
%   -batch_size:    (vector of int) batch size to test
% OUTPUTS:
%   -M_full:        (matrix of int) contains the budget used for every test.
%                   Along different row changes the leaning rate while the first column contains the results relative 
%                   to the GD/SGD and along the other changes the n_fine.
%                   The depth contains the different batch size (matrix lenght(batch_size) x length(learning_rate) x length(n_fine)+1)
%   -costi:         (matrix of double) contains the value of the cost function at the end of the training.
%                   The structure is equal to M_full (matrix lenght(batch_size) x length(learning_rate) x length(n_fine)+1)
%   -med_nCoarse:   (matrix of double) contains the mean number of correction done for every ParaFlowS test. 
%                   So has the same structure of costi_full and M_full but without the column related to GD/SGD 
%                   (matrix lenght(batch_size) x length(learning_rate) x length(n_fine))
%=======================================================================================================
    % Initialize the outputs matrices
    M_full=zeros(length(batch_size),length(learning_rate),length(n_fine)+1);
    costi_full=zeros(length(batch_size),length(learning_rate),length(n_fine)+1);
    med_nCoarse_full=zeros(length(batch_size),length(learning_rate),length(n_fine));
    
    % Initialize the global counters
    global iterGrad
    iterGrad=0;
    global iterFine
    iterFine=0;
    global iterCoarse
    iterCoarse=0;
    
    % Loop over the batch size
    for i=1:length(batch_size)
        disp("Start doing batch_fine/Gradient_batch="+string(batch_size(i)))
        data.batchsize_fine=batch_size(i);
        data.batchsize_gradient=batch_size(i);
        
        % Lauch the performance test for different batch size
        [M_full(i,:,:),costi_full(i,:,:),med_nCoarse_full(i,:,:)] = performanceTest(data,n_fine,learning_rate);

    end

end