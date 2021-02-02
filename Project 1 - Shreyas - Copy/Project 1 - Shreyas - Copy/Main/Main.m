clear all;
close all;
clc;

%importing all the output labels from the excel sheet

labs = readtable("trainLabels.csv");

result = ones(256,16);

%looping through all the files and generating individual matrices
%currently set at this number to visualize a .bytes files that has been
%visualized in the provided paper for reference of the grader

for i=1091:1091
    %constructing the file name and obtaining
    %Please change the path(where .bytes files are stored) below accordingly to obtain files
    str = strcat("C:/Users/turin/Desktop/train/"+table2array(labs(i,"Id"))+".bytes")
    %reading the file from local repository
    A = fileread(str);
    %returns the .bytes files as a matrix
    B = returnImage(A);
    %Visualising the created images
    figure,imagesc(B),title(table2array(labs(i,"Id"))+".bytes");
    colormap(gray);
    save("Created Images/"+i,"B");
end



