function [Tc,Tm,Cgrad,Ccoarse,Cfine,M_nCoarse]= TestRandomness(data)
    full_size=size(data.x,2);
    Cgrad=zeros(full_size,1);
    Ccoarse=zeros(full_size);
    Cfine=zeros(full_size);
    M_nCoarse=zeros(full_size,full_size);

    N=5; %use to set the number of run to make the mean

    global iterGrad
    iterGrad=0;
    global iterFine
    iterFine=0;
    global iterCoarse
    iterCoarse=0;

    for i=1:full_size
        disp("Start doing batch_fine/Gradient_batch="+string(i))
        data.batchsize_fine=i;
        data.batchsize_gradient=i;

        for k=1:N
            [~,~]=StocasticGradientDescent(data,false);
            Cgrad(i)=Cgrad(i)+iterGrad/N;
            iterGrad=0;
        end

        for j=1:full_size
            disp("Start doing batch_coarse="+string(j))
            data.batchsize_coarse=j;

            for k=1:N
                [costHistory,~] = paraflows(data,false);
                Ccoarse(i,j)=Ccoarse(i,j)+iterCoarse/N;
                iterCoarse=0;
                Cfine(i,j)=Cfine(i,j)+iterFine/N;
                iterFine=0;
                M_nCoarse(i,j)=M_nCoarse(i,j)+mean(costHistory(1,2:end))/N;
            end

        end
    end

    Tc=table((1:full_size)',Cgrad,'VariableNames',["batch_fine","GradientDescent"]);
    for i=1:full_size
      Tc=addvars(Tc,(string(Cfine(:,i))+','+string(Ccoarse(:,i))),'NewVariableNames',"batch_coarse="+string(i));
    end
    disp(Tc)

    Tm=table((1:full_size)','VariableNames',"batch_fine");
    for i=1:full_size
        Tm=addvars(Tm,M_nCoarse(:,i),'NewVariableNames',"batch_coarse="+string(i));
    end
    disp(Tm)

end