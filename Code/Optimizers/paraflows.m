function [costHistory,y1] = paraflows(data,stampa)
%=======================================================================================================
% Apply the ParaFlowS algorithm to the training of the neural network
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -stampa:        (bool) decide if show in the command window the training behaviour or not
% OUTPUTS:
%   -costHistory:   (matrix of double) values of the cost function
%   -y1:            (vector of double) weights and biases at the end of the training
%=======================================================================================================

n_fine=data.n_fine;
n_coarse=data.n_coarse;
dT=data.dT;
dt=data.dt;

% Set the seed for reproducibility
rng(50)
% Count the number of parameters
n_parameters=CountParameters(data);

y0=(1-2*rand(n_parameters,1));

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = y0;

costHistory = ones(1+n_fine+n_coarse,1);

% Compute the cost of the initial guess
if stampa
    costHistory(1) = costo(data,y0);
else
    [costHistory(1),~]=FandG(data,y0,randperm(size(data.x,2),data.batchsize_coarse));
end

% Update the global counter for testing
global iterCoarse
iterCoarse = 1;
global iterFine
iterFine = 0;

k=1;
convergence = true;

% ParaFlowS loop
while convergence
    U_coarse(:,1)=U_coarse_temp;

    tic;
    % Fine solver
    [fFine,costFine, U_fine] = fine_solver(U_coarse_temp, dt, n_fine,data,stampa);
    
    costHistory(2:n_fine+1,end)=costFine;

    m=0;
    errore=fFine(end);

    % Correction step
    [~,f, bff2] = coarse_solver(U_coarse_temp, dT,data,stampa);
    bff1=bff2;
    i=1;
    esci=true;
    % Continue to correct untill the cost function increses
    while esci && i<=n_coarse
        U_coarse(:,i+1) = bff1 + U_fine - bff2;
        [f,costHistory(1+n_fine+i,end), bff1] = coarse_solver(U_coarse(:,i+1), dT,data,stampa);
        
        if (f<errore || i==1)
            errore=f;
            m=i+1;
        else
            esci=false;
        end
        i=i+1;
    end

    time_iter = toc;

    costHistory=[costHistory ones(1+n_fine+n_coarse,1)];

    % Update the solution before restart the loop
    if m==0
        ParaflowError = norm(U_coarse_temp-U_fine,2);
        U_coarse_temp=U_fine;
        m=1;
    else
        ParaflowError = norm(U_coarse_temp-U_coarse(:,m),2);
        U_coarse_temp = U_coarse(:,m);
    end    
    costHistory(1,end)=m;
    k=k+1;
    
    % Print of the cost function behaviour (optional)
    if stampa && mod(k,200)==0
        disp(['iteration ' num2str(k) ', time: ', num2str(time_iter) ', cost_history = ', num2str(costHistory(n_fine+m-1,end-1)),' norm = ',num2str(ParaflowError)])
    end

    % Check the convergence
    budget = iterFine*data.batchsize_fine + iterCoarse*data.batchsize_coarse;
    
    convergence = costo(data,U_coarse_temp) > data.threshold && budget < data.Maxiter;
    %convergence = ParafloowError > 2.5*10^-6; % Other possible stopping citeria based on the difference between to subsequent iterates
end
y1=U_coarse_temp;

end