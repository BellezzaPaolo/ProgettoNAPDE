function [costHistory,y1] = parareal_system(data)
eta= data.eta;
MaxIter=data.Maxiter;
n_coarse=data.n_coarse;
n_parareal=data.n_parareal;

T = eta * MaxIter;
n_fine=ceil(MaxIter/n_coarse);
dT=T/n_coarse;
dt=dT/n_fine;

n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end

y0=0.5*randn(n_parameters,1);

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = zeros(n_parameters,n_coarse + 1);
U_fine = zeros(n_parameters,n_parareal);

U_coarse(:,1) = y0;
U_coarse_temp(:,1) = y0;

costHistory = zeros(n_coarse*n_fine,n_parareal);

% zeroth iteration
for i =1:n_coarse
    [f, y1] = coarse_solver((i-1)*dT, U_coarse_temp(:,i),dT,data);
    U_coarse_temp(:,i + 1) = y1;
end


% parareal loop
for k = 1:n_parareal
    tic;
    % parallel for (fine solver)
    parfor i = k:n_coarse
        [cost, y2] = fine_solver((i-1)*dT, U_coarse(:,i), dt, n_fine,data);
        U_fine(:,i)=y2;
        costFine(:,i)=cost;
    end
    gap=k;
    for jj=k:n_coarse
        costHistory(gap:gap+n_fine-1,k)=costFine(:,jj);
        gap=gap+n_fine;
    end

    % predict - correct
    for i = k:n_coarse
        [f, bff1] = coarse_solver((i-1)*dT, U_coarse(:,i), dT, data);
        [f, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(:,i), dT,data);
        U_coarse(:,i+1) = bff1 + U_fine(:,i) - bff2;
    end

    y1=U_fine(:,k);

    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 1e-4)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(n_coarse)])
        break
    end
    
    U_coarse_temp = U_coarse;
    
    % optional
    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(n_coarse) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_coarse-k)*time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', costHistory(end,k)])
end
end