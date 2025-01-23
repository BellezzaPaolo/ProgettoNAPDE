function [M,costi,med_nCoarse,Tm,Tc] = performanceTest(data,n_fine,learning_rate)
    M=zeros(length(learning_rate),length(n_fine)+1);
    costi=cell(length(learning_rate),length(n_fine)+1);
    global iterParaflow
    iterParaflow=0;

    for i=1:length(learning_rate)
        disp("Start doing learning_rate="+num2str(learning_rate(i)))
        data.eta=learning_rate(i);
        data.dT=learning_rate(i);
        data.dt=learning_rate(i);
        [costi{i,1},~]=StocasticGradientDescent(data,false);
        M(i,1)=iterParaflow;
        disp("Finisced gradient descent with " +int2str(iterParaflow)+" calls of FandG ");
        iterParaflow=0;
        for j=1:length(n_fine)
            data.n_fine=n_fine(j);
            [costi{i,j+1},~] = paraflows(data,false);
            M(i,j+1)=iterParaflow;
            disp("Finisced n_fine="+int2str(n_fine(j))+" with "+int2str(iterParaflow)+ " calls of FandG ");
            iterParaflow=0;
        end
        
    end

    Tc=table(learning_rate,M(:,1),'VariableNames',["learning_rate","GradientDescent"]);
    for i=1:length(n_fine)
        Tc=addvars(Tc,M(:,i+1),'NewVariableNames',"Paraflow fine="+int2str(n_fine(i)));
    end

    disp(Tc)

    med_nCoarse=zeros(size(costi,1),size(costi,2)-1);
    for i= 1:size(costi,1)
         for j=2:size(costi,2)
            med_nCoarse(i,j-1)=mean(costi{i,j}(1,2:end));
        end
    end
    
    Tm=table(learning_rate,'VariableNames',"learning_rate");
    for i=1:length(n_fine)
        Tm=addvars(Tm,med_nCoarse(:,i),'NewVariableNames',"Paraflow fine="+int2str(n_fine(i)));
    end

    disp(Tm)

end