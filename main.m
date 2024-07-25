clear
close all
clc


%==========================================================================
% LOAD DATA FOR TEST CASE
%==========================================================================

data= Dati('Test2');

%%
%==========================================================================
% CHECK THE CONVERGENCE OF THE FUNCTION FandG
%==========================================================================

clc
checkCorrectness(data);

%%
%==========================================================================
% TRAINING WITH SOCASTIC GRADIENT DESCENT
%==========================================================================

[costHistory,y]=StocasticGradientDescent(data);

semilogy(1:1e3:data.Maxiter,costHistory(1:1e3:end))

%% training with parareal using the old project
clc
N_fine=ceil(data.Maxiter/data.n_coarse);
T = data.eta * data.Maxiter;
n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end

y0=0.5*randn(n_parameters,1);
[U_lowest, k, costhistoryVec] = parareal_systems(T, data.n_coarse, N_fine, y0, data.x, data.y, data.sigma, data.sigmaprime, data.shape);
%%
%==========================================================================
% TRAINING WITH PARAREAL
%==========================================================================

[costHistory,y1] = parareal_system(data);

%%
%==========================================================================
% PLOT THE DATA
%==========================================================================

semilogy(1:size(costHistory,1),costHistory)
legend('iter1','iter2','iter3','iter4','iter5','iter6')


%%
%==========================================================================
% TRAINING WITH PARAREAL AND FLOATING 
%==========================================================================

clc
[costHistory,y1] = parareal_systemFloating(data);

