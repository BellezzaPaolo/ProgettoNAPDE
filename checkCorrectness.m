function [] = checkCorrectness(data)
%data è la struct con tutti i dati del problema come detto da colloquio.

% dichiarazione della funzione FandG: [f,df] = FandG(data,y0)
%dove f è la funzione valutata nel punto y0 ricevuto in
%ingresso e df il suo gradiente nel medesimo punto
n_parameters=0;
for l=1:data.L-1
    n_parameters=n_parameters+data.shape(l)*data.shape(l+1)+data.shape(l+1);
end

theta=rand(n_parameters,1);
h=rand(n_parameters,1);
eps=1;

for ii=1:4
    [fp,~]=FandG(data,theta+eps*h);
    [fm,~]=FandG(data,theta-eps*h);
    [~,df]=FandG(data,theta);
    %[~,df] = gradL(theta, data.x, data.y, data.sigma, data.sigmaprime, data.shape);
    %[fp,~] = gradL(theta+eps*h, data.x, data.y, data.sigma, data.sigmaprime, data.shape);
    %[fm,~] = gradL(theta-eps*h, data.x, data.y, data.sigma, data.sigmaprime, data.shape);
    err(ii)=abs((fp-fm)/(2.*eps)-df'*h);

    fprintf('eps %12d error   %d df*h %d\n',eps,err(ii),df'*h);
    %fprintf('theta+eps\n')
    %(theta+eps*h)'
    %fprintf('theta-eps*h\n')
    %(theta-eps*h)'

    eps=eps*0.1;
end
err
end