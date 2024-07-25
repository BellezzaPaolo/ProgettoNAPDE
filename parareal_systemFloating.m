function [costHistory,y1] = parareal_systemFloating(data)
eta= data.eta;
MaxIter=data.Maxiter;
n_coarse=data.n_coarse;
n_parareal=data.n_parareal;

T = eta * MaxIter;
n_fine=ceil(MaxIter/n_coarse);
dT=T/n_coarse;
dt=dT/n_fine;

n_parameters=CountParameters(data);

y0=0.5*randn(n_parameters,1);

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = zeros(n_parameters,n_coarse + 1);
U_fine = zeros(n_parameters,n_parareal);

costHistory = zeros(n_coarse*n_fine,n_parareal);


% parareal loop
for k = 1:n_parareal

    U_coarse(:,1) = y0;
    U_coarse_temp(:,1) = y0;
    for i =1:n_coarse
        [~, y1] = coarse_solver((i-1)*dT, U_coarse_temp(:,i),dT,data);
        U_coarse_temp(:,i + 1) = y1;
    end

    tic;
    % parallel for (fine solver)
    parfor i = 1:n_coarse
        [cost, y2] = fine_solver((i-1)*dT, U_coarse(:,i), dt, n_fine,data);
        U_fine(:,i)=y2;
        costFine(:,i)=cost;
    end
    gap=1;
    for jj=1:n_coarse
        costHistory(gap:gap+n_fine-1,k)=costFine(:,jj);
        gap=gap+n_fine;
    end

    % predict - correct
    fmin=Inf;
    for i = 1:n_coarse
        [~, bff1] = coarse_solver((i-1)*dT, U_coarse(:,i), dT, data);
        [~, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(:,i), dT,data);
        U_coarse(:,i+1) = bff1 + U_fine(:,i) - bff2;
        [f,~] = FandG(data,U_coarse(:,i+1));
        if f<=fmin
            fmin=f;
            i+1
            y0=U_coarse(:,i+1);
        end
    end

    y1=U_fine(:,k);

    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 1e-4)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(n_coarse)])
        break
    end

    
    % optional
    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(n_parareal) ', time: ', num2str(time_iter) ', time remaining: ', num2str((n_parareal-k)*time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,k))])
end
end