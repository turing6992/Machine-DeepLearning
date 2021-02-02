function [xNorm] = normalize_features(xInput)

%Calculating coloum-wise mean and subtracting it with the xi
xAndMeanDiff = (xInput - repmat(mean(xInput), size(xInput,1), 1));
%Dividing the result with std deviation to get normalized result
xNormAns = xAndMeanDiff./repmat(std(xInput), size(xInput,1), 1);

%Creating a matrix with a dimension to accomodate x0
xNorm = ones(size(xInput,1),size(xInput,2)+1);

xNorm(:,2:end) = xNormAns;

end

