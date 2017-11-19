% Load the file that contains the dataset
load('data_logistic.mat');
% Get the number of data samples in the dataset
n = size(z, 1);
% Find the randomly chosen indices for training, validation and test data.
% Each subset contains 70%, 15% and 15% of the original dataset
% respectively
[trainInd, valInd, testInd] = dividerand(n, 0.7, 0.15, 0.15);
% Merge validation and test indices
% 70% for training and 30% for test is sufficient
testInd = [testInd,valInd];
testInd = sort(testInd);
% Form the matrix that contains training features
trainSet = [z(trainInd,1), z(trainInd,2)];
% Form the matrix that contains test features
testSet = [z(testInd,1), z(testInd,2)];
% r vector that contains the associated classes for each training sample
r = z(trainInd,3);
% Learning rate
n = 1;
% Threshold vector
thresholdVector = [0.001; 0.001];

% d dimensional new weight vector
% Initialized to random numbers between -0.01 and 0.01
wNew = zeros(2,1);
for j=1:2
    wNew(j) = -1 + (1-(-1))*rand(1);
end

% d dimensional weight vector, initialized to 0
w = zeros(2,1);
% Count how many iterations are needed until convergence
iteration = 0; 
% Update weight vector until convergence
while abs(wNew - w) > thresholdVector
    iteration = iteration+1;
    for j=1:2
        w(j) = wNew(j);
    end
    % d dimensional delta weight vector
    wDelta = zeros(2,1);
    % For each sample in the training set
    for t=1:size(trainInd,2)
        out = 0;
        % d dimensional, j'th column of t'th sample.
        for j=1:2
            out = out + w(j) * trainSet(t,j);
        end
        % Sigmoid function for 2 classes
        y = 1/(1+ exp(-out));
        % Finding wDelta for each feature 
        for j= 1:2
            wDelta(j) = wDelta(j) + (r(t) - y)*trainSet(t,j);
        end
    end
    % Updating the weight vector as wNew
    % will be checked against the old weight vector v to detect convergence
    for j=1:2
        wNew(j)= wNew(j) + n*wDelta(j);
    end
end
% When the convergence criteria is satisfied, weight vector is updated
w = wNew;
% Class vector of the test set, initialized to 0
class = zeros(size(testInd,1));

% For every data sample in the test set, predict the class  
for t=1:size(testInd,2)
    out = transpose(w) * transpose(testSet(t,:));
    y = 1/(1+ exp(-out));
    % If y is greater than 0.5, 
    % this test sample will belong to the positive class
    if y >= 0.5
        class(t) = 1;
    % If y is less than 0.5, 
    % this test sample will belong to the negative class
    else
        class(t) = 0;
    end
end

% Plotting the result of the test set classification
X = [testSet(:,1), testSet(:,2)];
y = class;
pos = find(y == 1);
neg = find(y == 0);
figure; hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
legend('Class 0','Class 1')
hold off;
