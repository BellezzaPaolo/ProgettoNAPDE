function [df,f] = FandG(theta,data) %,x,yd,setJ,sig,d_sig,alpha,L,nn)
    global counter_full
    global counter_partial

    if length(data.setJ)==data.J
        counter_full = counter_full+1;
    else
        counter_partial = counter_partial+1;
    end

    x=data.x;
    yd=data.yd;
    setJ=data.setJ;
    sig=data.sig;
    d_sig=data.d_sig;
    alpha=data.alpha;
    L=data.L;
    nn=data.nn;

    % from theta assemble W
    ind=0;
    for k=2:length(nn)
        fact=nn(k-1)*nn(k);
        W{k} = reshape(theta(ind+1:ind+fact),nn(k),nn(k-1));
        ind = ind+fact;
    end
    for k=2:length(nn)
        fact = nn(k);
        b{k} = theta(ind+1:ind+fact);
        ind = ind+fact;
    end

    % compute cost and gradient
    [f,grad,z,delta] = F_layer(x,yd,setJ,sig,d_sig,W,b,alpha,1,L);
    % from grad assemble df
    df=0*theta;
    ind=0;
    for k=2:length(nn)
        fact=nn(k-1)*nn(k);
        df(ind+1:ind+fact) = reshape(grad{k},fact,1);
        ind = ind+fact;
    end
    for k=2:length(nn)
        fact = nn(k);
        df(ind+1:ind+fact) = grad{length(nn)+k};
        ind = ind+fact;
    end

end