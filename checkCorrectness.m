function [] = checkCorrectness(data)
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
    %[~, df] = gradL(theta, data.input_point, data.output_point, data.sigma, data.sigmaprime, data.shape);
    %[fp,~] = gradL(theta+eps*h, data.input_point, data.output_point, data.sigma, data.sigmaprime, data.shape);
    %[fm, ~] = gradL(theta-eps*h, data.input_point, data.output_point, data.sigma, data.sigmaprime, data.shape);
    err=abs((fp-fm)/(2.*eps)-df'*h);

    fprintf('eps %12d error   %d\n',eps,err);

    eps=eps*0.1;
end