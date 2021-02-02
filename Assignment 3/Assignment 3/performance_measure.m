function [accuracy,confusion_matrix,Precision,recall,f1Score,specificity] = performance_measure(y,yCap)

accuracy = size(find(y==yCap),1);

truePositives = size(find(y==1 & yCap==1),1);

falsePositives = size(find(y==0 & yCap==1),1);

falseNegatives = size(find(y==1 & yCap==0),1);

trueNegatives = size(find(y==0 & yCap==0),1);

confusion_matrix = [truePositives falsePositives;falseNegatives trueNegatives]

Precision = (truePositives)/(truePositives + falsePositives);

recall = (truePositives)/(truePositives + falseNegatives);

f1Score = (2 * Precision * recall)/(Precision+recall);

specificity = trueNegatives/(trueNegatives+falsePositives);

end

