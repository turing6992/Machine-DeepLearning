clc;
clear all;
close all;

%loading the malware data set
data = load("Malware_Dataset");

%normalizing the features of the total image dataset
x = normalize_features(data.X);
y = data.y;

%No of PCA features to consider
N = 10;

%No of folds to be performed
k = 10;

%arrays to store the output values perfold
OAy = ones(1,1);
Opl = ones(1,1);

%obtain train and test indices for all the k folds
for idx=1:k
test_indices{idx}=idx:k:10868;
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

%Template svm creates kernel to study various types of svms such as linear gaussian and quadratic 
%quadratic being a type of polynomial can be created by varing the
%polynomial degree parameter
t = templateSVM('Standardize',true,'KernelFunction','polynomial');
M = fitcecoc(Y,y,'Learners',t);

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
