clear
%close all
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

[costHistory,y]=StocasticGradientDescent(data,true);


%==========================================================================
% PLOT THE  ERROR
%==========================================================================

graficoErrore(data,costHistory,'Stocastic')


%==========================================================================
% PLOT THE CLASSIFICATION REGION
%==========================================================================

Disegno(data,y)

%%
%==========================================================================
% TRAINING WITH PARAREAL
%==========================================================================

[costHistory,y] = parareal_system(data);

%%
%==========================================================================
% PLOT THE  ERROR
%==========================================================================

graficoErrore(data,costHistory,'Parareal')

%%
%==========================================================================
% PLOT THE CLASSIFICATION REGION
%==========================================================================

Disegno(data,y)

%%
%==========================================================================
% PLOT THE DATA
%==========================================================================
figure()
for ii=1:size(costHistory,2)
    subplot(4,5,ii)
    semilogy(1:size(costHistory,1),costHistory(:,ii))
    legend('iter'+string(ii))
end


%%
%==========================================================================
% TRAINING WITH PARAFLOWS 
%==========================================================================

%clc
[costHistory,y] = paraflows(data,true);

%%
%==========================================================================
% PLOT THE  ERROR
%==========================================================================

graficoErrore(data,costHistory,'Paraflow')

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
%close all
n_coarse=6;
n_fine=2000;

figure()
semilogy(1,costHistory(1,1),'o');
gap=0;
c=['#0072BD';'#D95319';'#EDB120';'#7E2F8E';'#77AC30';'#4DBEEE';'#A2142F'];
hold on
for ii=1:size(costHistory,2)-1
    semilogy(gap+2:gap+n_fine+1,costHistory(2:n_fine+1,ii),'Color',c(mod(ii-1,size(c,2))+1,:));
    semilogy(gap+n_fine+2:n_fine:gap+n_fine*costHistory(1,ii+1)+n_fine+1,costHistory(n_fine+2:n_fine+2+costHistory(1,ii+1)),'o','MarkerEdgeColor',c(mod(ii-1,size(c,2))+1,:));
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

learning_rate=[0.1;0.01];
n_fine=[10,100];
[M,costi,M_nCoarse,Tm,Tc] = performanceTest(data,n_fine,learning_rate);


%%
clc

figure(1)
surf(learning_rate,n_fine,M(:,2:end)','EdgeColor','none','FaceColor','interp');
xscale('log')
yscale('log')
colorbar
clim([min(min(M(:,2:end))) max(max(M(:,2:end)-10^5))]);
ylabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')

figure(2)
subplot(2,2,1)
loglog(learning_rate,M(:,2:end)')
hold on
loglog(learning_rate,M(:,1)','LineWidth',4)
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')
ylabel('number of calls')
subplot(2,2,2)
loglog(learning_rate,mean(M(:,2:end)',1))
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')
ylabel('mean number of calls')
subplot(2,2,3)
loglog(n_fine,M(:,2:end))
xlabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
ylabel('number of calls')
hold on
loglog(n_fine,M(:,1),'LineWidth',4)
subplot(2,2,4)
loglog(n_fine,mean(M(:,2:end),1))
xlabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
ylabel('mean number of calls')

%% analisys of the number of nCoarse

figure(1)
s=surf(learning_rate,n_fine,med_nCoarse','EdgeColor','none','FaceColor','interp');
colorbar
clim([min(min(med_nCoarse)) max(max(med_nCoarse-120))]);
ylabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')

figure(2)
subplot(2,2,1)
plot(learning_rate,med_nCoarse')
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')
ylabel('mean nCoarse')
title('mean number over the iterations of coarse pass done for every  n_fine')
subplot(2,2,2)
plot(learning_rate,mean(med_nCoarse,2))
xlabel('learning rate ['+string(learning_rate(1))+','+string(learning_rate(end))+']')
ylabel('mean nCoarse')
title('total mean number of coarse pass done')
subplot(2,2,3)
plot(n_fine,med_nCoarse)
xlabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
ylabel('mean nCoarse')
title('mean number of coarse pass done for every  learning rate')
subplot(2,2,4)
plot(n_fine,mean(med_nCoarse,1))
xlabel('n fine ['+string(n_fine(1))+','+string(n_fine(end))+']')
ylabel('mean nCoarse')
title('totatl mean number of coarse pass done for every  learning rate')

%%
%==========================================================================
% PERFORMANCE TEST WITH RANDOMICITY
%==========================================================================
clc

[Tc,Tm,Cgrad,Ccoarse,Cfine,M_nCoarse]= TestRandomness(data);

