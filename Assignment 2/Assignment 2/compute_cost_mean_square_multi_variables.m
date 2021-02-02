function [J] = compute_cost_mean_square_multi_variables(thetas,x,y)
    
    J = (1/(2*size(x,1))) * sum((x*thetas' - y).^2);    

end

