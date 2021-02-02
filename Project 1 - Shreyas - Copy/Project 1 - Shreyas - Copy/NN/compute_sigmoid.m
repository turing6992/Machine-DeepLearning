function [g] = compute_sigmoid(z,flag)

    %Calculating sigmoid value for every element of the O*X_Norm
    g = (1./(1+exp(-z)));
    if flag == 1
        g = g.*(1-g);
    end
end

