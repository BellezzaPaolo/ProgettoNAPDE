function [M,costi,med_nCoarse] = performanceTest(data,n_fine,learning_rate)
%=======================================================================================================
% Test the efficiency of the ParaFlowS algorithm respect to SGD or GD
% varing the learning rate (data.eta and data.dt) and number of refinements
% associated to the fine operator (data.n_fine)
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -n_fine:        (vector of int) n_fine to test
%   -learning_rate: (vector of double) learning rate to test
% OUTPUTS:
%   -M:             (matrix of int) contains the budget used for every test. 
%                   Along different row changes the leaning rate while the first column contains the results relative 
%                   to the GD/SGD and along the other changes the n_fine (matrix length(learning_rate) x length(n_fine)+1)
%   -costi:         (matrix of double) contains the value of the cost function at the end of the training.
%                   The structure is equal to M (matrix length(learning_rate) x length(n_fine)+1)
%   -med_nCoarse:   (matrix of double) conatins the mean number of correction done for every ParaFlowS test. 
%                   So has the same structure of costi and M but without the column related to GD/SGD 
%                   (matrix length(learning_rate) x length(n_fine))
%=======================================================================================================
    % Initialize output matrices
    M=zeros(length(learning_rate),length(n_fine)+1);
    costi=zeros(length(learning_rate),length(n_fine)+1);
    med_nCoarse=zeros(length(learning_rate),length(n_fine));

    % Intialize the global counters
    global iterGrad
    iterGrad=0;
    global iterFine
    iterFine=0;
    global iterCoarse
    iterCoarse=0;

    % Loop over the different learning rate
    for i=1:length(learning_rate)
        disp("Start doing learning_rate="+num2str(learning_rate(i)))
        data.eta=learning_rate(i);

        % test GD/SGD
        [~,y]=StocasticGradientDescent(data,false);
        % Save data and print the progresses
        costi(i,1) = costo(data,y);
        M(i,1)=iterGrad*data.batchsize_gradient;
        disp("Finisced gradient descent with " +int2str(M(i,1))+" calls of FandG and error= "+string(costi(i,1)));
        iterGrad=0;

        % Loop over the different n_fine
        for j=1:length(n_fine)
            data.n_fine=n_fine(j);
            data.dt=learning_rate(i);
            data.dT=data.dt*data.n_fine;

            % test ParaFlowS
            [a,y] = paraflows(data,false);
            % Save data and print the progresses
            med_nCoarse(i,j)=mean(a(1,2:end));
            costi(i,j+1) = costo(data,y);
            M(i,j+1)=iterFine*data.batchsize_fine+iterCoarse*data.batchsize_coarse;
            disp("Finisced n_fine="+int2str(n_fine(j))+" with "+int2str(M(i,j+1))+ " calls of FandG and error= "+string(costi(i,j+1))+" and num="+string(iterCoarse));
            iterFine=0;
            iterCoarse=0;
        end
        
    end
end