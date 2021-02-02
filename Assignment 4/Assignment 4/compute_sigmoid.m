function [g] = compute_sigmoid(z)

    %Calculating sigmoid value for every element of the O*X_Norm
    g = (1./(1+exp(-z)));

end

