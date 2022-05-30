clear all
clc

seed = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(seed);

%wine_quality_red_R;
%waveform_R;
%For mat file
load iris.mat
load labels.mat
load folds.mat


dataX = iris;
dataY = labels;
test_indx = logical(folds(:,1));
train_indx = logical(1-test_indx);

U_dataY = unique(dataY);
nclass = numel(U_dataY);
dataY_temp = zeros(numel(dataY),nclass);

% 0-1 coding for the target
for i=1:nclass
    idx = dataY==U_dataY(i);
    dataY_temp(idx,i)=1;
end

dataX = rescale(dataX);

trainX = dataX(train_indx,:);
trainY = dataY_temp(train_indx,:);
testX = dataX(test_indx,:);
testY = dataY_temp(test_indx,:);

train_data = [trainX,trainY];
test_data = [testX,testY];

[~, ~, TrainingAccuracy, TestingAccuracy] = OSELM(train_data, test_data, 1, 100, 'sig', 100, 100);
TrainingAccuracy
TestingAccuracy