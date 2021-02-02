function [ts,cts] = gradient_descent_lr_one_variable(thetas,x,y,alpha,numOfIterations)

    t0 = thetas(1);
    t1 = thetas(2);
    
    costs = zeros(1,numOfIterations);
    
    theta0PrevStep = t0;
    theta1PrevStep = t1;
    
    for i=1:numOfIterations
        t0 = t0 - (alpha * tCal([theta0PrevStep,theta1PrevStep],x,y,0));
        t1 = t1 - (alpha * tCal([theta0PrevStep,theta1PrevStep],x,y,1));
        
        theta0PrevStep = t0;
        theta1PrevStep = t1;
        
        costs(1,i) = run_compute_cost_mean_square([t0,t1],x,y);
    end
    cts = costs;
    ts = [t0,t1];
end

% Calculating the value of htheta and subtracting it with yi
%based on theta1 or theta 0 multiplying it with xi

function [thetaCal] = tCal(thetas,x,y,val)

sumtheta = 0;

    if val == 0
        for i=1:size(x,2)
           sumtheta = sumtheta + (thetas(1)+(thetas(2)*x(i)) - y(i));
        end
    else
        for i=1:size(x,2)
           sumtheta = sumtheta + (thetas(1)+(thetas(2)*x(i)) - y(i))*x(i);
        end
    end
    thetaCal = (1/size(x,2))* sumtheta;
end