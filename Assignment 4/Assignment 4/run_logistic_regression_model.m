clear all;
close all;
clc;
data = load("Sample Data.txt");

%splitting the data i.e getting the inputs separated from the output
x = data(:,1:end-1);
y = data(:,end);

colors = y;
colors(colors==0) = 'r';
colors(colors==1) = 'b';

scatter(x(:,1),x(:,2),[],colors),ylabel('FeatureX2'),xlabel('FeatureX1');

% normalizing the value of x wrt mean and standard deviation and adding 1
% to X0
xNorm = normalize_features(x);

%initializing theta values as prescribed
thetas = zeros(1,size(x,2)+1);

%cost computation function
%[J] = compute_cost_logistic_regression(thetas,xNorm,y,lambda);

%total iterations
numberIts = 2000;
%learning rate
alpha = 0.1;

lambda = [0 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 1000];

for i=1:size(lambda,2)

%Applying gradient descent
[costs,ts] = gradient_descent_logistic_regression_regularized(thetas,xNorm,y,numberIts,alpha,lambda(i));

figure,plot(1:numberIts,costs),title("Learning Curve For Given Values"),ylabel('Costs'),xlabel('Iteration #');


%Predicting the value
yCap = prediction(ts,xNorm)';
disp(" ");
disp("Lambda: "+lambda(i));

disp("The Thetas ");
disp(ts);

%final Answers

[accuracy,confusion_matrix,Precision,recall,f1Score,specificity] = performance_measure(y,yCap);

disp("The accuracy is "+accuracy+"%");

disp("Precision: "+Precision);

disp("Recall: "+recall);

disp("F1 Score: "+f1Score);

disp("Specificity: "+specificity);

end





