function [f,df] = FandG(data,y0)

    x=data.x;
    y=data.y;
    shape=data.shape;
    L=data.L;
    sigma=data.sigma;
    sigmaprime=data.sigmaprime;
    index=randperm(size(x,2),data.batchsize);

    % compose the set of matrices and weights from the vector y0
    pointer_y0 = 1;
    for l = 1:(L-1)
        W{l} = zeros(shape(l+1), shape(l));
        % assign every row of the matrix
        gap = shape(l)-1;
        for i = 1:shape(l+1)
            W{l}(i,:) = y0(pointer_y0 : pointer_y0 + gap)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        % assign bias vector
        gap = shape(l+1)-1;
        b{l} = y0(pointer_y0:pointer_y0 + gap);    % column vector
        pointer_y0 = pointer_y0 + gap + 1;
    end
    
    % compute the derivative of the Loss function w.r.t every wheight
    % initialize structures
    z = cell(1,L-1);
    delta = cell(1, L-1);
    a = cell(1,L);
   
    
    % forward pass
    a{1} = x(:,index);
    for l = 1:L-1
       z{l} = W{l}*a{l}+b{l};
       a{l+1} = sigma(z{l});
       % D{l}=diag(sigmaprime(z{l}));
    end
    
    
    % backward pass
    delta{L-1} = sigmaprime(z{L-1}).*(a{L}-y(:,index));
    for l = (L-1):-1:2
        delta{l-1} = sigmaprime(z{l-1}).*(W{l}'*delta{l});
    end
    
    
    % update weights and bias
    for l = 1:L-1
        W{l} = delta{l}*(a{l})';
        b{l} = mean(delta{l},2);
    end


    
    % now transform back again the matrices into a vector
    % prepare initial conditions (unfold matrices and biases)
    
    y1=zeros(size(y0,1),1);

    % recycle memory of y0
    pointer_y0 = 1;
    for l = 1:(L-1)
        gap = shape(l) - 1;
        for jj = 1:size(W{l},1)
            y1(pointer_y0:pointer_y0 + gap ) = W{l}(jj,:)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        gap = shape(l+1)-1;
        y1(pointer_y0:pointer_y0 + gap) =  b{l};
        pointer_y0 = pointer_y0 + gap + 1;
    end
    
    f=0.5.*mean((a{end}-y(:,index)).^2,'all');

    df=-y1;

end