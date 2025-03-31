%==========================================================================
% CLEAR THE WORKSPACE AND ADD PATHS
%==========================================================================
clear
close all
clc

addpath Optimizers
addpath Postprocessing
addpath Testing
addpath Utilities

%%

%==========================================================================
% LOAD DATA FOR TEST CASE
%==========================================================================

data= Dati('TestHigham');

%%
%==========================================================================
% CHECK THE CONVERGENCE OF THE FUNCTION FandG
%==========================================================================

clc
checkCorrectness(data);

%%
%==========================================================================
% TRAINING WITH STOCASTIC GRADIENT DESCENT AND PLOT THE RESULT
%==========================================================================

clc
[costHistoryS,yS]=StocasticGradientDescent(data,true);

graficoErrore(data,costHistoryS,'Stocastic')

Disegno(data,yS)

%%
%==========================================================================
% TRAINING WITH PARAREAL AND PLOT THE RESULT
%==========================================================================
clc

[costHistory,y] = parareal_system(data,true);

graficoErrore(data,costHistory,'Parareal')

Disegno(data,y)

%%
%==========================================================================
% TRAINING WITH PARAFLOWS AND PLOT THE RESULT
%==========================================================================

clc
[costHistory,y] = paraflows(data,true);

graficoErrore(data,costHistory,'Paraflow')

Disegno(data,y)

%%
%==========================================================================
% PERFORMANCE TEST
%==========================================================================
clc

learning_rate=[1e0; 1e-1; 1e-2; 1e-3];
n_fine=[10, 50, 100, 500, 1000, 2000];

[M,costi,M_nCoarse] = performanceTest(data,n_fine,learning_rate);


%%
%==========================================================================
% PERFORMANCE TEST WITH RANDOMICITY
%==========================================================================
clc

learning_rate=[1e0; 1e-1; 1e-2; 1e-3];
n_fine=[10, 50, 100, 500, 1000, 2000];
batch_size = [1 30];

[costi_full,med_nCoarse_full,M_full]= TestRandomness(data,learning_rate,n_fine,batch_size);
