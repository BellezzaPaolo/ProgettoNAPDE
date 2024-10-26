function [costHistory,y1] = paraflows(data)
%=======================================================================================================
% Apply the paraflows algorithm to the training of the NN
%=======================================================================================================
% INPUTS:
%   -data: struct that cointans every parameters of the problem (see Dati.m)
%        bias of the NN
% OUTPUTS:
%   -costHistory: matrix that contains the values of the Loss function
%   -y1: vector that contains the weights and biases at the end of the
%       training
%=======================================================================================================
L=data.L;

%T = eta * MaxIter;
%n_fine=ceil(MaxIter/n_coarse);
n_fine=2000;
n_coarse=6;
dT=0.01;%T/n_coarse;
dt=0.01;%T/(n_fine);

rng(50)
% count the number of parameters
n_parameters=CountParameters(data);

y0=(1-2*rand(n_parameters,1));

U_coarse = zeros(n_parameters,n_coarse + 1);

%[~,U_coarse(:,1)] = coarse_solver(0, y0, dT,data);
U_coarse_temp = y0;

costHistory = ones(1+n_fine+n_coarse,1);
[costHistory(1),~]=FandG(data,y0,randperm(size(data.x,2),data.batchsize_coarse));

k=1;
errore=100;
while errore>10^-3 && k<100
    U_coarse(:,1)=U_coarse_temp;
    % parareal loop
    tic;
    % fine solver
    [costFine, U_fine] = fine_solver(0*dT, U_coarse_temp, dt, n_fine,data);
    
    costHistory(2:n_fine+1,end)=costFine;
    %a=zeros(n_coarse,1); b=zeros(n_coarse,1);

    m=100;
    % predict - correct
    for i = 1:n_coarse
        [~, bff1] = coarse_solver((i-1)*dT, U_coarse(:,i), dT,data);
        [~, bff2] = coarse_solver((i-1)*dT, U_coarse_temp, dT,data);
        % a(i)=norm(bff1-bff2);
        % b(i)=norm(U_fine-bff2);
        U_coarse(:,i+1) = bff1 + U_fine - bff2;
        
        [costHistory(1+n_fine+i,end),~]=FandG(data,U_coarse(:,i+1),randperm(size(data.x,2),data.batchsize_coarse));

        if costHistory(1+n_fine+i,end)<=errore
            errore=costHistory(1+n_fine+i,end);
            m=i+1;
        end
    end
    
    % m=100;
    % for c=1:n_coarse
    % 
    %     [costHistory(1+n_fine+c,end),~]=FandG(data,U_coarse(:,c+1),randperm(size(data.x,2),data.batchsize_coarse));
    % 
    %     if costHistory(1+n_fine+c,end)<=errore
    %         errore=costHistory(1+n_fine+c,end);
    %         m=c+1;
    %     end        
    % end

    if k>=10
        %keyboard
    end
    % time and print (optional)
    time_iter = toc;
    disp(['iteration ' num2str(k) ', time: ', num2str(time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,end))])
    costHistory=[costHistory ones(1+n_fine+n_coarse,1)];
    if m==100
        U_coarse_temp=U_fine;
        m=1;
    else
        U_coarse_temp = U_coarse(:,m);
    end    
    costHistory(1,end)=m;
    k=k+1;
end
y1=U_coarse_temp;
errore
k
end