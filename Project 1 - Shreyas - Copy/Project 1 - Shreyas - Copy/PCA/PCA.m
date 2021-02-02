clc;
clear all;
close all;

data = load("Malware_Dataset");

x = normalize_features(data.X);
y = data.y;
N = 10;

k = 10;

OAy = ones(1,1);
Opl = ones(1,1);

for idx=1:10
test_indices{idx}=idx:10:10868;
train_indices{idx}=setdiff(1:10868,test_indices{idx});
end


for i=1:k
    
    validX = x(test_indices{1,i},:);
%Sotring the corresponding labels for validation
    validY = y(test_indices{1,i});

%removing the indeices obtained from previous step from the main data set
%to obtain traininng dataset
x(test_indices{1,i},:) = [];
y(test_indices{1,i}) = [];

disp("fold"+i)

%passing the data set and N value for obtaining the transformation function
[A,Y,d] = PCA_transformation(x,N);

%Calulating transform using the valdation data set and A
Y_Valid = validX*A;

%using fitch KNN to classify the nearset neighbors 
M = fitcknn(Y,y,'NumNeighbors',1,'DistanceWeight','squaredinverse','NSMethod','euclidean');

%using predict function to predict the output with the trained model and
%validation matrix
predicted_labels = predict(M,Y_Valid);
OAy = [OAy;validY];
Opl = [Opl;predicted_labels];

x = normalize_features(data.X);
y = data.y;
    
end


OAy(1,:) = [];
Opl(1,:) = [];

validY = ind2vec(OAy'); 

predicted_labels = ind2vec(Opl');

figure,plotconfusion(validY,predicted_labels)
