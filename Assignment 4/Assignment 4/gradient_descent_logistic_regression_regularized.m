function [costs,ts] = gradient_descent_logistic_regression_regularized(thetas,xNorm,y,numberIts,alpha,lambda)
        
    %saving the theta values to a temporary variables
    oldThetas = thetas;
    %empty array for costs
    cts = zeros(numberIts,1);
    for i = 1:numberIts
        %calculating theta value
        thetas(1) = thetas(1) - alpha*((1/length(y))*sum((compute_sigmoid(xNorm*oldThetas')-y).*xNorm(:,1)));
        thetas(2:end) = (thetas(2:end)*(1-((alpha*lambda)/length(y)))) - alpha*((1/(length(y)))*sum((compute_sigmoid(xNorm*oldThetas')-y).*xNorm(:,2:end)));
        %reassigning for the loop
        oldThetas(1) = thetas(1);
        oldThetas(2:end) = thetas(2:end);
        %Cost calculation
        cts(i) = compute_cost_logistic_regression_regularized(oldThetas,xNorm,y,lambda);

    end

%return Variables    
ts = thetas;
costs = cts;



end

