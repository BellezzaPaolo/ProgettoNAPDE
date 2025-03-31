function [costHistory,y1] = parareal_system(data,stampa)
%=======================================================================================================
% Apply the Parareal algorithm to the training of the neural network
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -stampa:        (bool) decide if show in the command window the training behaviour or not
% OUTPUTS:
%   -costHistory:   (matrix of double) values of the cost function
%   -y1:            (vector of double) weights and biases at the end of the training
%=======================================================================================================

n_parareal=data.n_coarse;
n_fine = data.n_fine;
n_coarse = data.n_coarse;
dT = data.dT;
dt = data.dt;


% Count the number of parameters
n_parameters=CountParameters(data);
% Set the seed for reproducibility
rng(50)
y0=0.5*randn(n_parameters,1);

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = zeros(n_parameters,n_coarse + 1);
U_fine = zeros(n_parameters,n_parareal);

U_coarse(:,1) = y0;
U_coarse_temp(:,1) = y0;

costHistory = zeros(n_coarse*n_fine,n_parareal);

% Zeroth iteration
for i =1:n_coarse
    [f,fExact, y0] = coarse_solver( U_coarse_temp(:,i),dT,data,stampa);
    U_coarse_temp(:,i + 1) = y0;
end


% Parareal loop
for k = 1:n_parareal
    tic;

    % Parallel for (fine solver)
    parfor i = k:n_coarse
        [f,cost, y0] = fine_solver(U_coarse_temp(:,i), dt, n_fine,data,stampa);
        U_fine(:,i)=y0;
        costFine(:,i)=cost;
    end
    gap=1;
    for jj=1:n_coarse
        costHistory(gap:gap+n_fine-1,k)=costFine(:,jj);
        gap=gap+n_fine;
    end

    % Correction step
    for i = k:n_coarse
        [~,~, bff2] = coarse_solver(U_coarse_temp(:,i), dT,data,stampa);
        U_coarse(:,i+1) = U_coarse_temp(:,i + 1) + U_fine(:,i) - bff2;
    end

    y1=U_fine(:,end);

    % Check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 0.32)
        time_iter = toc;
        disp(['iteration ' num2str(k) '/' num2str(n_parareal) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_parareal-k)*time_iter)])
        disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,k)),', norm  =  ', num2str(incr)])
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(n_coarse)])
        break
    end
    
    U_coarse_temp = U_coarse;
    
    % Time and print (optional)
    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(n_parareal) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_parareal-k)*time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,k)),', norm  =  ', num2str(incr)])
end
end