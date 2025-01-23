function [f] = costo(data,y0)
L = data.L;
sigma=data.sigma;
pointer_y0=0;
% assign weights matrices
for ii=2:L
    W{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii-1)*data.shape(ii)),data.shape(ii),data.shape(ii-1));
    pointer_y0=pointer_y0+data.shape(ii-1)*data.shape(ii);
end
% assign bias vector
for ii=2:L
    b{ii}=reshape(y0(pointer_y0+1:pointer_y0+data.shape(ii)),data.shape(ii),1);
    pointer_y0=pointer_y0+data.shape(ii);
end
f = 0;
for ii = 1:size(data.x,2)
    a{1}=data.x(:,ii);%j);
    for l=2:L
        z{l}=W{l}*a{l-1}+b{l};
        a{l}=sigma(z{l});
    end

    % compute the cost function
    f=f+0.5*norm(a{end}-data.y(:,ii)).^2;
end