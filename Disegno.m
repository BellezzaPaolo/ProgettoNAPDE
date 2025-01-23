function []=Disegno(data,y1)
L=data.L;
pointer_y0=0;
for ii=2:L
    W{ii}=reshape(y1(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii)),data.shape(ii),data.shape(ii-1));
    pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
end

% assign bias vector
for ii=2:L
    b{ii}=reshape(y1(pointer_y0+1:pointer_y0+data.shape(ii)),data.shape(ii),1);
    pointer_y0=pointer_y0+data.shape(ii);
end

if strcmp(data.name,'TestHigham')
    N = 500;
    Dx = 1/N;
    Dy = 1/N;
    xvals = [0:Dx:1];
    yvals = [0:Dy:1];
    L=data.L;
    x=data.x;
    sig=data.sigma;
    for k1 = 1:N+1
        xk = xvals(k1);
        
        for k2 = 1:N+1
            yk = yvals(k2);
            a{1} = [xk;yk];
            for l=2:L
                a{l} = sig(W{l}*a{l-1}+b{l});
            end
            
            Aval(k2,k1) = a{L}(1);
            Bval(k2,k1) = a{L}(2);
        
        end
    end
    figure()
    [X,Y] = meshgrid(xvals,yvals);
    clf
    Mval = Aval>Bval;
    contourf(X,Y,Mval,[0.5 0.5])
    hold on
    colormap([1 1 1; 0.8 0.8 0.8])
    plot(x(1,1:floor(size(x,2)/2)),x(2,1:floor(size(x,2)/2)),'ro','MarkerSize',12,'LineWidth',4)
    plot(x(1,floor(size(x,2)/2)+1:size(x,2)),x(2,floor(size(x,2)/2)+1:size(x,2)),'bx','MarkerSize',12,'LineWidth',4)
    
    xlim([0,1])
    ylim([0,1])
else
    if strcmp(data.name,'TestSeno')
        f=@(x)  sin(3.*x);
        xdis=linspace(0,pi,100);
    elseif strcmp(data.name,'TestPolinomio')
        f=@(x) 4.*x.^3-4.*x.^2+2;
        xdis=linspace(-1,1,100);
    elseif strcmp(data.name,'TestSmorzato')
        f=@(x) exp(-x).*sin(x);
        xdis=linspace(0,10,100);
    end
    
    ypred=ones(1,length(xdis));
    a=cell(1,data.L);
    for i=1:length(xdis)
        a{1}=xdis(i);
        for l=2:data.L
            a{l}=data.sigma(W{l}*a{l-1}+b{l});
        end
        ypred(i)=a{end};
    end

    figure()
    plot(data.x,data.y,'*')
    hold on
    plot(xdis,f(xdis),'-')
    plot(xdis,ypred,'-')
    legend('batch point','real function','approximated function')
    title(data.name)
    
  
end