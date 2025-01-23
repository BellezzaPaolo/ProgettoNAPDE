function [costHistory,y1] = paraflows(data,stampa)
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

n_fine=data.n_fine;
n_coarse=data.n_coarse;
dT=data.dT;
dt=data.dt;

rng(50)
% count the number of parameters
n_parameters=CountParameters(data);

y0=(1-2*rand(n_parameters,1));

U_coarse = zeros(n_parameters,n_coarse + 1);

%[~,U_coarse(:,1)] = coarse_solver(0, y0, dT,data);
U_coarse_temp = y0;

costHistory = ones(1+n_fine+n_coarse,1);
[costHistory(1),~]=FandG(data,y0,randperm(size(data.x,2),data.batchsize_coarse));
global iterCoarse
iterCoarse=iterCoarse+1;

k=1;
errore=90;
while errore>10^-3 && k<data.Maxiter %&& errore<10^2
    U_coarse(:,1)=U_coarse_temp;
    % parareal loop
    tic;
    % fine solver
    [costFine, U_fine] = fine_solver(0*dT, U_coarse_temp, dt, n_fine,data);
    
    costHistory(2:n_fine+1,end)=costFine;
    %a=zeros(n_coarse,1); b=zeros(n_coarse,1);

    m=0;
    errore=costHistory(n_fine+1,end);
    % predict - correct
    [~, bff2] = coarse_solver((1-1)*dT, U_coarse_temp, dT,data);
    bff1=bff2;
    i=1;
    esci=true;
    while esci && i<=n_coarse
        U_coarse(:,i+1) = bff1 + U_fine - bff2;
        [costHistory(1+n_fine+i,end), bff1] = coarse_solver((i-1)*dT, U_coarse(:,i+1), dT,data);
        

        if costHistory(n_fine+i+1,end)<=errore || i==1
            errore=costHistory(1+n_fine+i,end);
            m=i+1;
        else
            esci=false;
        end
        i=i+1;
    end

    
    % time and print (optional)
    time_iter = toc;
    if stampa && mod(k,200)==0
        disp(['iteration ' num2str(k) ', time: ', num2str(time_iter) ', cost_history = ', num2str(costHistory(n_fine+m-1,end))])
    end
    costHistory=[costHistory ones(1+n_fine+n_coarse,1)];
    if m==0
        U_coarse_temp=U_fine;
        m=1;
    else
        U_coarse_temp = U_coarse(:,m);
    end    
    costHistory(1,end)=m;
    k=k+1;
end
y1=U_coarse_temp;

%if errore>100
%   print("parareal not converged")
%end

end