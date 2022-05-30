load stock.mat
data_all = normc(table2array(stockportfolioperformancedatasetS4));

rand_sequence=randperm(size(data_all,1));           
temp_data=data_all;
data_all=temp_data(rand_sequence, :); 

Nsample = size(data_all,1);
Nfea = size(data_all,2)-1;
train_index = floor(Nsample*0.7);
trainX = data_all(1:train_index,1:Nfea);
testX = data_all(train_index+1:Nsample,1:Nfea);
trainY = data_all(1:train_index,Nfea+1);
testY = data_all(train_index+1:Nsample,Nfea+1);
nclass = numel(trainY);

c1 = cvpartition(train_index,'KFold',5);

for i=1:5
cvtrainx{i} = trainX(training(c1,i),:);
cvtrainy{i} = trainY(training(c1,i),:);
cvtestx{i} = trainX(test(c1,i),:);
cvtesty{i} = trainY(test(c1,i),:);
end