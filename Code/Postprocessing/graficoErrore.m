function [] = graficoErrore(data,costHistory,type)
%=======================================================================================================
% Plot the value of the cost function during training
%=======================================================================================================
% INPUTS:
%   -data:          (struct) contains every parameters of the problem (see Dati.m)
%   -costHistory:   (N-dimesional vector of double) values of the cost
%                   function during training. The way it is organized
%                   depends on the optimization tchnic used
%   -type:          (string) type of optimization technic has been used
%=======================================================================================================
if strcmp(type,'Stocastic')
    figure()
    % Vector of times
    t = linspace(0,size(costHistory,2)*data.eta,size(costHistory,2));
    % Plot the vactor of costs
    semilogy(t,costHistory,'LineWidth',4)

    %title('value of the loss function using the stocastic gradient')
    %axis([0 3e4 0.001 3])
    xlabel('time','FontSize',14)
    ylabel('cost function L ','FontSize',14)

elseif strcmp(type,'Parareal')
    % Vector of times on the coarse grid and on the finer one
    t_coarse = 0:data.dT:(data.n_coarse-1)*data.dT;
    t_fine = 0:data.dt:(data.n_fine-1)*data.dt;
    cost_coarse = zeros(1,data.n_coarse);

    figure()

    % Computes the number of times Parareal algorithm has iterated
    Nsub = sum(not(costHistory(1,:)==0));

    iter = 1;
    % Iterate over the n_coarse subintervals
    for jj = 1:data.n_fine:size(costHistory,1)
        cost_coarse(iter) = costHistory(jj,Nsub);
        % Plot the costs of that subinterval
        a = semilogy(t_fine+t_coarse(iter),costHistory(jj:jj+data.n_fine-1,Nsub),'LineWidth',4);
        a.Color = [0, 0.4470, 0.7410];
        hold on
        iter = iter +1;
    end
    % Add the costs ralated to the coarse operator
    %a = semilogy(t_coarse,cost_coarse,'*'); commented because n_coarse too big and so this is meaningless
    %a.Color = [0, 0.4470, 0.7410];
    title('iter '+string(Nsub))

    xlabel('time')
    ylabel('cost function L ')

elseif strcmp(type,'Paraflow')
    figure()
    
    % Vector of times relative of the fine solver
    t_fine = 0:data.dt:data.dt*(data.n_fine-1);
    tgap = 0;
    % Plot the initial cost value
    semilogy(0,costHistory(1,1),'b')
    % Loop over the itarate of the ParaFlowS
    for jj = 1:size(costHistory,2)-1
        % Add the costs associated to that iteration
        hold on
        a = semilogy(tgap + t_fine,costHistory(2:data.n_fine+1,jj),'LineWidth',4);
        a.Color = [0, 0.4470, 0.7410];

        m = min(costHistory(1,jj+1)+1,data.n_coarse+1);
        % Compute the vecto fo the coarse times
        t_coarse = tgap+t_fine(end)+data.dT:data.dT:tgap+t_fine(end)+data.dT*(m-1);
        % Add the costs of the coarse predictions
        a = semilogy(t_coarse,costHistory(data.n_fine+2:data.n_fine+m,jj),'.','MarkerSize',8);%'LineWidth',4);
        a.Color= [0, 0.4470, 0.7410];
        
        tgap = t_coarse(end);
    end

    %title('value of the loss function using the parareal')
    axis([0 3e4 0.001 3])
    xlabel('time','FontSize',14)
    ylabel('cost function L ','FontSize',14)
end
end