function [cost,grad,z,delta] = F_layer(x,yd,setJ,sig,d_sig,W,b,alpha,L1,L2)
    cost = 0;
    for ell=L1+1:L2
        cost = cost + 0.5*alpha*norm(W{ell},'fro')^2;
        grad{ell} = alpha*W{ell};
    end
    for ell=L1+1:L2
        cost = cost + 0.5*alpha*norm(b{ell})^2;
        grad{L2+ell} = alpha*b{ell};
    end
    J=length(setJ);
    for jj=1:J
        j=setJ(jj);
        y{L1,j} = x{j};
        for ell=L1+1:L2
            z{ell,j} = W{ell}*y{ell-1,j}+b{ell};
            y{ell,j} = sig(z{ell,j});
        end
        cost = cost + 0.5*norm(yd{j}-y{L2,j})^2;
        if nargout>2
            delta{L2,j} = d_sig(z{L2,j}).*(y{L2,j}-yd{j});
            for ell=L2-1:-1:L1+1
                delta{ell,j} = d_sig(z{ell,j}).*(W{ell+1}'*delta{ell+1,j});
            end
            for ell=L1+1:L2
                grad{ell}=grad{ell}+(delta{ell,j}*y{ell-1,j}');
            end
            for ell=L1+1:L2
                grad{L2+ell} = grad{L2+ell}+delta{ell,j};
            end
        end
    end
end