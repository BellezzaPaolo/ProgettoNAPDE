clear
close all
clc


%==========================================================================
% LOAD DATA FOR TEST CASE
%==========================================================================

data= Dati('TestSmorzato');

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

[costHistory,y]=StocasticGradientDescent(data,true);

semilogy(1:1e3:data.Maxiter,costHistory(1:1e3:end))

%%
%==========================================================================
% TRAINING WITH PARAREAL
%==========================================================================

[costHistory,y] = parareal_system(data);

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
% TRAINING WITH PARAFLOWS 
%==========================================================================

%clc
[costHistory,y] = paraflows(data,true);

%%
close all
n_coarse=6;
n_fine=2000;
costHistorydis=NaN(size(costHistory,1)*size(costHistory,2),size(costHistory,2)-1);
figure()
for i=1:size(costHistory,2)-1
    costHistorydis(i*n_coarse*n_fine:i*n_coarse*n_fine+n_coarse*n_fine-1,i)=costHistory(:,i);
end

%hold on
semilogy(costHistorydis(1:100:end,:))
grid on

%%
close all
n_coarse=6;
n_fine=2000;

figure()
semilogy(1,costHistory(1,1),'o');
gap=0;
c=['#0072BD';'#D95319';'#EDB120';'#7E2F8E';'#77AC30';'#4DBEEE';'#A2142F'];
hold on
for ii=1:size(costHistory,2)-1
    semilogy(gap+2:gap+n_fine+1,costHistory(2:n_fine+1,ii),'Color',c(mod(ii-1,size(c,2))+1,:));
    semilogy(gap+n_fine+2:n_fine:gap+n_fine*n_coarse+n_fine+1,costHistory(n_fine+2:end,ii),'o','MarkerEdgeColor',c(mod(ii-1,size(c,2))+1,:));
    gap=costHistory(1,ii+1)*n_fine+gap;
end

%%
%==========================================================================
% PLOT THE CLASSIFICATION REGION
%==========================================================================

Disegno(data,y)

%%
%==========================================================================
% PERFORMANCE TEST
%==========================================================================
clc

[M,costi,T] = performanceTest(data);
