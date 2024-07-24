function [cost,y0] = fine_solver(t0, y0, dt, n_fine,data)
cost=zeros(n_fine,1);
for ii=1:n_fine
    [f,df]=FandG(data,y0);
    y0=y0+dt*df;
    cost(ii)=f;
end
end