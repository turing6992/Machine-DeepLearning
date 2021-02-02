clear all;
close all;
clc;

% Input Data
 x = [0.4,0.56,0.6,0.8,1];
 % Expected values
 y = [11, 14, 15, 22, 26];
 
 %weights for calculation
 theta0 = 0.5;
 theta1 = 1;
 
 % making it as an error
 thetas = [theta0,theta1];
 
 %learning rate
 alpha = 0.4;
 
 %Number of iterations to train weights
 numOfIterations = 300;
 
 %Function to just test costs - can be uncommented
 %Jtheta = compute_cost_mean_square(thetas,x,y);
 
 %passing base theta value, input,expected output, learning rate and n
 %umber of iterations
 [ts,costs] = gradient_descent_lr_one_variable(thetas,x,y,alpha,numOfIterations);
 
 %Plotting the costs to number of iterations
 figure,plot(1:numOfIterations,costs),title("Learning Curve"),ylabel('Costs'),xlabel('Iteration #');
 
 disp("The results after 300 iterations");
 
 disp("Theta 0");
 disp(ts(1));
 disp("Theta 1");
 disp(ts(2));
 disp("Minimized cost");
 disp(costs(size(costs,2)));
 
 
 