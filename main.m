clear
close all
clc


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
% TRAINING WITH STOCASTIC GRADIENT DESCENT
%==========================================================================

[costHistory,y]=StocasticGradientDescent(data);

semilogy(1:1e3:data.Maxiter,costHistory(1:1e3:end))

%%
%==========================================================================
% TRAINING WITH PARAREAL
%==========================================================================

[costHistory,y1] = parareal_system(data);

%%
%==========================================================================
% PLOT THE DATA
%==========================================================================
figure()
c=['iter1';'iter2';'iter3';'iter4';'iter5';'iter6';'iter7';'iter8';'iter9';'ite10';'ite11';'ite12'];
for ii=1:12
    subplot(4,3,ii)
    semilogy(1:size(costHistory,1),costHistory(:,ii))
    legend(c(ii,:))
end


%%
%==========================================================================
% TRAINING WITH PARAREAL AND FLOATING 
%==========================================================================

clc
[costHistory,y1] = paraflows(data);


%%
close all
n_coarse=20;
n_fine=2000;
costHistorydis=NaN(size(costHistory,1)*size(costHistory,2),size(costHistory,2)-1);
figure()
for i=1:size(costHistory,2)-1
    costHistorydis(i*n_coarse*n_fine:i*n_coarse*n_fine+n_coarse*n_fine-1,i)=costHistory(:,i);
end

%hold on
semilogy(costHistorydis(1:100:end,:))
grid on
