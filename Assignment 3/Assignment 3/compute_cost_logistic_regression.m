function [J] = compute_cost_logistic_regression(thetas,x_norm,y)

    %passing thetas and normalized features to sigmoid function
    h = compute_sigmoid(thetas*x_norm');
    %calculating cost
    J = (-1/size(y,1))*(sum((y.*log(h'))+((1-y).*log(1-h'))));
    
end

