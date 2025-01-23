function [] = graficoErrore(data,costHistory,type)
if strcmp(type,'Stocastic')
    figure()

    t = linspace(0,size(costHistory,2)*data.eta,size(costHistory,2));

    semilogy(t,costHistory)

    title('value of the loss function using the stocastic gradient')
    xlabel('time')
    ylabel('loss function L ')

elseif strcmp(type,'Parareal')
    t_coarse = 0:data.dT:(data.n_coarse-1)*data.dT;
    t_fine = 0:data.dt:(data.n_fine-1)*data.dt;
    cost_coarse = zeros(1,data.n_coarse);

    figure()
    Nsub = sum(not(costHistory(1,:)==0));

    for ii = 1:Nsub
        subplot(Nsub/5,5,ii)
        iter = 1;
        for jj = 1:data.n_fine:size(costHistory,1)
            cost_coarse(iter) = costHistory(jj,ii);
            semilogy(t_fine+t_coarse(iter),costHistory(jj:jj+data.n_fine-1,ii),'LineWidth',4)
            hold on
            iter = iter +1;
        end
        %semilogy(t_coarse,cost_coarse,'*')
        title('iter '+string(ii))
    end

    %title('value of the loss function using the parareal')
    %xlabel('time')
    %ylabel('loss function L ')

elseif strcmp(type,'Paraflow')
    figure()

    t_fine = 0:data.dt:data.dt*(data.n_fine-1);
    tgap = 0;
    semilogy(0,costHistory(1,1),'*')
    for jj = 1:size(costHistory,2)-1

        hold on
        semilogy(tgap + t_fine,costHistory(2:data.n_fine+1,jj),'LineWidth',8)

        m = min(costHistory(1,jj+1)+1,data.n_coarse+1);
        t_coarse = tgap+t_fine(end)+data.dT:data.dT:tgap+t_fine(end)+data.dT*(m-1);
        
        semilogy(t_coarse,costHistory(data.n_fine+2:data.n_fine+m,jj),'*','MarkerSize',1) % m vale?
        
        tgap = t_coarse(end);
    end

    title('value of the loss function using the parareal')
    xlabel('time')
    ylabel('loss function L ')
end
end