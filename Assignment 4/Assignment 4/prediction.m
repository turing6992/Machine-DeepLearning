function [predictions] = prediction(thetas,x_norm)

    %Applying the modified theta values to xNorm to predict outputs
    predictions = compute_sigmoid(thetas*x_norm');
    
    %Assigning 0 if the value is below treshold
    predictions(predictions < 0.5) = 0;
    
    %Assigning 1 if the value is above treshold
    predictions(predictions >= 0.5) = 1;
    
end

