clear all
clc

b = rand(1,100);
c = repmat(b,99,1);
size(c)
size(c')

w = 2*rand(11,100);%-1;


%break down train data into batches of 100
%wine_quality_red_R;
plant_margin_R;
data=data(:,2:end);
dataX=data(:,1:end-1);

dataY=data(:,end);

U_dataY = unique(dataY);
nclass = numel(U_dataY);
dataY_temp = zeros(numel(dataY),nclass);

% 0-1 coding for the target
for i=1:nclass
    idx = dataY==U_dataY(i);
    dataY_temp(idx,i)=1;
end

trainX=dataX;
trainY=dataY_temp;
testX=dataX;
testY=dataY_temp;

[Nsample,Nfea] = size(trainX);
 %X = randn(31,1518);
 k = 1;
 for nn = 1:100:Nsample
    batch_trainx{k} = trainX(nn:nn+99,:);
    batch_trainY{k} = trainY(nn:nn+99,:);
    k = k+1;
 end