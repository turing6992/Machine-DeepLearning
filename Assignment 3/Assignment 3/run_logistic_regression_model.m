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
[J] = compute_cost_logistic_regression(thetas,xNorm,y);

%total iterations
numberIts = 1000;
%learning rate
alpha = 0.2;

%Applying gradient descent
[costs,ts] = gradient_descent_logistic_regression(thetas,xNorm,y,numberIts,alpha);

figure,plot(1:numberIts,costs),title("Learning Curve For Given Values"),ylabel('Costs'),xlabel('Iteration #');


%Predicting the value
yCap = prediction(ts,xNorm)';

disp("The Thetas ");
disp(ts);

%final Answers

[accuracy,confusion_matrix,Precision,recall,f1Score,specificity] = performance_measure(y,yCap)

disp("The accuracy is "+accuracy+"%");

disp("The Number of true positives: "+confusion_matrix(1));

disp("The Number of false positives: "+confusion_matrix(3));

disp("The Number of false negatives: "+confusion_matrix(2));

disp("The Number of true negatives: "+confusion_matrix(4));

disp("Precision: "+Precision);

disp("Recall: "+recall);

disp("F1 Score: "+f1Score);

disp("Specificity: "+specificity);







