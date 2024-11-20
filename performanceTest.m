function [M,costi,T] = performanceTest(data)
    learning_rate=[0.1; 0.05; 0.01; 0.005; 0.001; 0.0005];
    n_fine=[10; 50; 100; 300; 500; 650; 750; 850; 1000; 3000; 5000];
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
        disp("Finisced gradient descent with" +int2str(iterParaflow)+"calls of FandG ");
        iterParaflow=0;
        for j=1:length(n_fine)
            data.n_fine=n_fine(j);
            [costi{i,j+1},~] = paraflows(data,false);
            M(i,j+1)=iterParaflow;
            disp("Finisced n_fine="+int2str(n_fine(j))+" with "+int2str(iterParaflow)+ "calls of FandG ");
            iterParaflow=0;
        end
        
    end

    T=table(learning_rate,M(:,1),'VariableNames',["learning_rate","GradientDescent"]);
    for i=1:length(n_fine)
        T=addvars(T,M(:,i+1),'NewVariableNames',"Paraflow fine="+int2str(n_fine(i)));
    end

    disp(T)

end