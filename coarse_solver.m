function [f,y1]=coarse_solver(t0,y0,dT,data)
    [f,df] = FandG(data,y0);
    y1=y0+dT*df;
end