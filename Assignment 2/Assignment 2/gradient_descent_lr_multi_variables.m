function [costs,ts] = gradient_descent_lr_multi_variables(thetas,xNorm,y,numberIts,alpha)
        
    %saving the theta values to a temporary variables
    oldThetas = thetas;
    %empty array for costs
    cts = zeros(numberIts,1);
    
    for i = 1:numberIts
        %calculating theta value
        thetas = thetas - alpha*((1/length(y))*sum(((xNorm*oldThetas')-y).*xNorm));
        oldThetas = thetas;
        cts(i) = compute_cost_mean_square_multi_variables(oldThetas,xNorm,y);

    end
        
ts = thetas;
costs = cts;


end

