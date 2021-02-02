clear all;
close all;
clc;
data = load("Body Fat.txt");

%splitting the data i.e getting the inputs separated from the output
x = data(:,1:end-1);
y = data(:,end);

% normalizing the value of x wrt mean and standard deviation and adding 1
% to X0
xNorm = normalize_features(x);

%initializing theta values as prescribed
thetas = zeros(1,size(x,2)+1);

%cost computation function
%[J] = compute_cost_mean_square_multi_variables(thetas,xNorm,y);

%number or iterations
numberIts = 300;
%learning Rate
alpha = 0.05;

%calculating gradient discent 
[costs,ts] = gradient_descent_lr_multi_variables(thetas,xNorm,y,numberIts,alpha);

%plotting the values 
figure,plot(1:numberIts,costs),title("Learning Curve For Body Fat"),ylabel('Costs'),xlabel('Iteration #');
 
disp("Values for Body Fat");

disp("Final Theta Values");
disp(ts);
disp("Final Cost");
disp(costs(end));

% Part 2 for Air Foil
%The only change from the above code is the file name 

data = load("airfoil_self_noise.txt");

x = data(:,1:end-1);
y = data(:,end);

xNorm = normalize_features(x);

thetas = zeros(1,size(x,2)+1);

[J] = compute_cost_mean_square_multi_variables(thetas,xNorm,y);

numberIts = 300;
alpha = 0.05;

[costs,ts] = gradient_descent_lr_multi_variables(thetas,xNorm,y,numberIts,alpha);

figure,plot(1:numberIts,costs),title("Learning Curve For Air foil Self Noise"),ylabel('Costs'),xlabel('Iteration #');
 
disp("Values for Air foil noise");

disp("Thetas");
disp(ts);
disp("Final Cost");
disp(costs(end));



