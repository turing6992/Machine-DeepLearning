close all;
clear all;
clc;

matfiles = dir('Created Images/*.mat');
labs = readtable("trainLabels.csv");

dataSet = ones(10868,256*16);

for i=1:1
    string = strcat("Created Images/"+matfiles(i).name);
    matfile = load(string);
    matfile.B(:)'
end

