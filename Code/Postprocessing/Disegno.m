function []=Disegno(data,y1)
%=======================================================================================================
% Plot the predictions and the training set in order to check the quality
% of the predictions
%=======================================================================================================
% INPUTS:
%   -data:      (struct) contains every parameters of the problem (see Dati.m)
%   -y1:        (vector of double) weights and biases of the neural network
%=======================================================================================================
L=data.L;
pointer_y0=0;

% Assign weight matrices
for ii=2:L
    W{ii}=reshape(y1(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii)),data.shape(ii),data.shape(ii-1));
    pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
end
% Assign bias vector
for ii=2:L
    b{ii}=reshape(y1(pointer_y0+1:pointer_y0+data.shape(ii)),data.shape(ii),1);
    pointer_y0=pointer_y0+data.shape(ii);
end
% Check which test is used
if strcmp(data.name,'TestHigham')
    % Computes the N values to predict
    N = 500;
    Dx = 1/N;
    Dy = 1/N;
    xvals = [0:Dx:1];
    yvals = [0:Dy:1];
    L=data.L;
    x=data.x;
    sig=data.sigma;
    % Forward pass of the network for every value
    for k1 = 1:N+1
        xk = xvals(k1);
        
        for k2 = 1:N+1
            yk = yvals(k2);
            a{1} = [xk;yk];
            for l=2:L
                a{l} = sig(W{l}*a{l-1}+b{l});
            end
            
            % Update the matrices that will distinct the 2 predicted regions
            Aval(k2,k1) = a{L}(1);
            Bval(k2,k1) = a{L}(2);
        
        end
    end


    figure()
    % Compute the mesh
    [X,Y] = meshgrid(xvals,yvals);
    clf
    % Denote the 2 predicted regions with 0 or 1 and plot them
    Mval = Aval>Bval;
    contourf(X,Y,Mval,[0.5 0.5])
    hold on
    colormap([1 1 1; 0.8 0.8 0.8])
    % Add the training points to the plot
    plot(x(1,1:floor(size(x,2)/2)),x(2,1:floor(size(x,2)/2)),'ro','MarkerSize',12,'LineWidth',4)
    plot(x(1,floor(size(x,2)/2)+1:size(x,2)),x(2,floor(size(x,2)/2)+1:size(x,2)),'bx','MarkerSize',12,'LineWidth',4)
    
    xlim([0,1])
    ylim([0,1])
    title(data.name)

elseif strcmp(data.name,'TestSmorzato')
    f=@(x) exp(-x).*sin(x);
    % set the values to predict
    xdis=linspace(0,10,100);
    ypred=ones(1,length(xdis));

    % Forward pass of the network
    a=cell(1,data.L);
    for i=1:length(xdis)
        a{1}=xdis(i);
        for l=2:data.L
            a{l}=data.sigma(W{l}*a{l-1}+b{l});
        end
        ypred(i)=a{end};
    end

    figure()
    % Plot training set
    plot(data.x,data.y,'*')
    hold on
    % Add the real function
    plot(xdis,f(xdis),'-')
    % Add the predicted function
    plot(xdis,ypred,'-')
    legend('batch point','real function','approximated function')
    title(data.name)
    
end