clear all;
close all;
clc;

%importing dataset
load('Malware_Dataset.mat');

%normalizing the data
x = normalize_features(X);
%assigning the ouput labels
y = y;

%radomizing values to split data set into 80/20 for testing and validation
R = randperm(10868);
indices = R(1:2172);

%assigning validation indices
validX = x(indices,:);
validY = y(indices);

%removing all the validation indices from the main dataset to use it as
%training data set
x(indices,:) = [];
y(indices) = [];

L = 2;

%Number of nuerons in the 1st and second layer respectively
s2 = 400;
s3 = 300;

%number of classes
k = 9;

%Number of iterations
numOfItrs = 10000;
%Learning rate
alpha = 0.2;
%theta values based on the number of nuerons
thetaOne = (rand(s2,4098) * (2*0.12) - 0.12);
thetaTwo = (rand(s3,s2+1) * (2*0.12) - 0.12);
thetaThree = (rand(k,s3+1) * (2*0.12) - 0.12);

%Passing the data through hidden layers 
[J,thetaOne,thetaTwo,thetaThree]=nn_two_hidden_layers(thetaOne,thetaTwo,thetaThree,x,y,numOfItrs,alpha);

%Predicting the output based on the trained theta values
[predictedClass] = determine_output(thetaOne, thetaTwo,thetaThree,validX);

%calculating the accuracy
accuracy=length(find(predictedClass-validY'==0))/length(validY);

validY = ind2vec(validY');
predictedClass = ind2vec(predictedClass);
figure,plotconfusion(validY,predictedClass);

 