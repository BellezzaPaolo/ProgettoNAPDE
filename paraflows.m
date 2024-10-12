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
n_coarse=20;
dT=0.01;%T/n_coarse;
dt=0.01;%T/(n_fine);

% count the number of parameters
n_parameters=CountParameters(data);

y0=0.5*randn(n_parameters,1);

U_coarse = zeros(n_parameters,n_coarse + 1);
U_coarse_temp = zeros(n_parameters,n_coarse + 1);
U_fine = zeros(n_parameters,n_coarse);

U_coarse(:,1) = y0;
U_coarse_temp(:,1) = y0;

costHistory = ones(n_coarse*n_fine,1);

diff=1;
k=1;
errore=100;
while errore>10^-3 && diff>5*10^-3 && k<10^2
    % zeroth iteration
    for i =1:n_coarse
        [~, y0] = coarse_solver((i-1)*dT, U_coarse_temp(:,i),dT,data);
        U_coarse_temp(:,i + 1) = y0;
    end
    
    % parareal loop
    tic;
    % parallel for (fine solver)
    parfor i = 1:n_coarse
        [cost, y0] = fine_solver((i-1)*dT, U_coarse_temp(:,i), dt, n_fine,data);
        U_fine(:,i)=y0;
        costFine(:,i)=cost;
    end
    gap=1;
    for jj=1:n_coarse
        costHistory(gap:gap+n_fine-1,end)=costFine(:,jj);
        gap=gap+n_fine;
    end

    % predict - correct
    for i = 1:n_coarse
        [~, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(:,i), dT,data);
        U_coarse(:,i+1) = U_coarse_temp(:,i + 1) + U_fine(:,i) - bff2;
    end
    
    m=1;
    for c=1:n_coarse
        W=cell(L,1);
        b=cell(L,1);
        a=cell(L,1);
        
        % compose the set of matrices and weights from the vector y0
        pointer_y0=0;
        % assign weights matrices
        for ii=2:L
            W{ii}=reshape(U_coarse(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii),c),data.shape(ii),data.shape(ii-1));
            pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
        end
        % assign bias vector
        for ii=2:L
            b{ii}=reshape(U_coarse(pointer_y0+1:pointer_y0+data.shape(ii),c),data.shape(ii),1);
            pointer_y0=pointer_y0+data.shape(ii);
        end
        
        f=0;           
        
        for j=1:size(data.x,2)
            % forward pass
            a{1}=data.x(:,j);
            for l=2:L
                a{l}=data.sigma(W{l}*a{l-1}+b{l});
            end
            
            % compute the cost function
            f=f+0.5*norm(a{end}-data.y(:,j)).^2;
        end
        

        if f<=errore
            disp('entrato')
            errore=f;
            m=c;
        end        
    end
    
    m
    diff=norm(U_coarse_temp(:,1)-U_coarse(:,m));
    U_coarse_temp(:,1) = U_coarse(:,m);
    
    % time and print (optional)
    time_iter = toc;
    disp(['iteration ' num2str(k) ', time: ', num2str(time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(costHistory(end,end))])
    costHistory=[costHistory ones(n_coarse*n_fine,1)];
    k=k+1;
end
y1=U_coarse_temp(:,1);
errore
k
diff
end