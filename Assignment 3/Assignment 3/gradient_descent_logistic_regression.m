function [costs,ts] = gradient_descent_logistic_regression(thetas,xNorm,y,numberIts,alpha)
        
    %saving the theta values to a temporary variables
    oldThetas = thetas;
    %empty array for costs
    cts = zeros(numberIts,1);
    for i = 1:numberIts
        %calculating theta value
        thetas = thetas - alpha*((1/length(y))*sum((compute_sigmoid(xNorm*oldThetas')-y).*xNorm));
        %reassigning for the loop
        oldThetas = thetas;
        %Cost calculation
        cts(i) = compute_cost_logistic_regression(oldThetas,xNorm,y);

    end

%return Variables    
ts = thetas;
costs = cts;



end

