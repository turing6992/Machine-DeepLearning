function [J] = run_compute_cost_mean_square(thetas,x,y)

Jtheta = 0;
%Calculating costs
for i=1:size(x,2)
    hTheta = (thetas(1) + (thetas(2) *x(i)));
    Jtheta = Jtheta + (hTheta-y(i))^2;
end

Jtheta = (1/(2*size(x,2))) * Jtheta;

J = Jtheta;
end

