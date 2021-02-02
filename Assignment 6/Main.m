close all;
clear all;
clc;

%loading the data set
data = load("Sample_MNIST.mat");
x = data.X;
y = data.y;

%Using display data function to visualise a random set of 100 samples
displayData(x(501:600,:));

%using randperm to generate random values to remove data points from the data
%set
R = randperm(5000);
indices = R(1:1000);
%storing the removed data points for validation
validX = x(indices,:);
%Sotring the corresponding labels for validation
validY = y(indices);

%removing the indeices obtained from previous step from the main data set
%to obtain traininng dataset
x(indices,:) = [];
y(indices) = [];

%visualizing the validation dataset
displayData(validX);

%Number of top features to be considered
N = 20;

%passing the data set and N value for obtaining the transformation function
[A,Y,d] = PCA_transformation(x,N);

%Calulating transform using the valdation data set and A
Y_Valid = validX*A;

%using fitch KNN to classify the nearset neighbors 
M = fitcknn(Y,y,'NumNeighbors',10,'DistanceWeight','squaredinverse','NSMethod','euclidean');

%using predict function to predict the output with the trained model and
%validation matrix
predicted_labels = predict(M,Y_Valid);

%Calculating accuracy for cross reference
accuracy = ((size(find(validY==predicted_labels),1))/size(validY,1))*100;
disp("Accuracy is "+accuracy+"%");

disp("Plotting");

%converting using ind2vec for faster results with the plot confusion
%function
validY = ind2vec(validY');
predicted_labels = ind2vec(predicted_labels');

figure,plotconfusion(validY,predicted_labels)

