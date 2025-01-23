function [costHistory,y1] = parareal_system(data)
%=======================================================================================================
% Apply the parareal algorithm to the training of the NN
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
%        bias of the NN
% OUTPUTS:
%   -costHistory: matrix that contains the values of the Loss function
%   -y1: vector that contains the weights and biases at the end of the
%       training
%=======================================================================================================
eta= data.eta;
MaxIter=data.Maxiter;
n_parareal=data.n_coarse;
n_fine = data.n_fine;
n_coarse = data.n_coarse;
dT = data.dT;
dt = data.dt;

%T = eta * MaxIter;
%n_fine=ceil(MaxIter/n_coarse);
%T/n_coarse;
%T/(n_fine);

% count the number of parameters
n_parameters=CountParameters(data);
rng(50)
y0=0.5*randn(n_parameters,1);

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = zeros(n_parameters,n_coarse + 1);
U_fine = zeros(n_parameters,n_parareal);

U_coarse(:,1) = y0;
U_coarse_temp(:,1) = y0;

costHistory = zeros(n_coarse*n_fine,n_parareal);

% zeroth iteration
for i =1:n_coarse
    [f, y0] = coarse_solver((i-1)*dT, U_coarse_temp(:,i),dT,data);
    U_coarse_temp(:,i + 1) = y0;
end


% parareal loop
for k = 1:n_parareal
    tic;
    % parallel for (fine solver)
    parfor i = k:n_coarse
        [cost, y0] = fine_solver((i-1)*dT, U_coarse_temp(:,i), dt, n_fine,data);
        U_fine(:,i)=y0;
        costFine(:,i)=cost;
    end
    gap=1;
    for jj=1:n_coarse
        costHistory(gap:gap+n_fine-1,k)=costFine(:,jj);
        gap=gap+n_fine;
    end

    % predict - correct
    for i = k:n_coarse
        %[~, bff1] = coarse_solver((i-1)*dT, U_coarse(:,i), dT, data);
        [~, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(:,i), dT,data);
        U_coarse(:,i+1) = U_coarse_temp(:,i + 1) + U_fine(:,i) - bff2;
    end

    y1=U_fine(:,end);

    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);%/norm(U_coarse,2);
    if (incr < 1e-3 || k == 20)
        time_iter = toc;
        disp(['iteration ' num2str(k) '/' num2str(n_parareal) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_parareal-k)*time_iter)])
        disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,k)),'  =  ', num2str(incr)])
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(n_coarse)])
        break
    end
    
    U_coarse_temp = U_coarse;
    
    % time and print (optional)
    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(n_parareal) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_parareal-k)*time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,k)),'  =  ', num2str(incr)])
end
end