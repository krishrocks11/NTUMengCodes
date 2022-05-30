clear all
clc

seed = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(seed);
%%%DEMO
%{
load Car.mat
load labels.mat
load folds.mat


dataX = car;
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


%default values. you need to tune them for best results
option.N = 100;
option.L = 10;
option.C = 2^(4);
option.scale = 1;
option.activation = 2;
option.renormal = 1;
option.normal_type = 1;
[model,train_acc,test_acc] = MRVFL(trainX,trainY,testX,testY,option);

%% ACC for layer 1 to L
train_acc
test_acc
%}

option1.type = 1; % 0 for Class 1 for Reg
%Shiva CLASS Test
%{
%For mat file

load abalone.mat
%load haberman_survival.mat
%load iris.mat
%load pendigits.mat %large dataset (10k+)
%load pima.mat
%load plant_margin.mat
%load seeds.mat
%load waveform.mat
%load wine.mat
%load wine_quality_red.mat
%load wine_quality_white.mat
load labels.mat
load folds.mat

dataX = abalone;
%dataX = haberman_survival;
%dataX = iris;
%dataX = pendigits;
%dataX = pima;
%dataX = plant_margin;
%dataX = seeds;
%dataX = waveform;
%dataX = wine;
%dataX = wine_quality_red;
%dataX = wine_quality_white;
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


%For direct m file
%{
%wine_quality_red_R;
%seeds_R;
%iris_R;
waveform_R;
%pima_R;
%haberman_survival_R;
data=data(:,2:end);
dataX=data(:,1:end-1);

%{
% do normalization for each feature
mean_X=mean(dataX,1);
dataX=dataX-repmat(mean_X,size(dataX,1),1);
norm_X=sum(dataX.^2,1);
norm_X=sqrt(norm_X);
norm_X=repmat(norm_X,size(dataX,1),1);
dataX=dataX./norm_X;
%}

dataY=data(:,end);

U_dataY = unique(dataY);
nclass = numel(U_dataY);
dataY_temp = zeros(numel(dataY),nclass);

% 0-1 coding for the target
for i=1:nclass
    idx = dataY==U_dataY(i);
    dataY_temp(idx,i)=1;
end

%dataX = rescale(dataX); %Why not needed?

trainX=dataX;
trainY=dataY_temp;
testX=dataX;
testY=dataY_temp;
%}

%GRID SEARCH CLASS

% settings
%m1 is L, m2 is N, 3 is n for C, 4 is b1size, 5 is olbsize
m1 = 9; m2 = 3; m3 = 10;
%m4 = 6; %For small datasets
%m4 = 5; %For large datasets
m5 = 6;
nR = 5; % number of grid refinememnts
int_length_10 = 10; %interval lengths
int_length_12 = 12;
int_length_6 = 6;
int_length_30 = 30;
int_length_50 = 50;
int_length_100 = 100;
int_length_1000 = 1000;
%construct initial grid
G1 = linspace(2,int_length_10,m1);
G2 = linspace(8,int_length_10,m2);%For small datasets
%G2 = linspace(10,int_length_12,m2);%For large datasets
G3 = linspace(-12,int_length_6,m3);
%G4 = linspace(50,int_length_100,m4);%For small datasets
%G4 = linspace(100,int_length_100,m4);%For large datasets
G5 = linspace(20,int_length_30,m5);%For small datasets
%G5 = linspace(50,int_length_100,m5);%For large datasets

%initial value, must be very small
x1_max = -1e+10;
x2_max = -1e+10;
x3_max = -1e+10;
%x4_max = -1e+10;
x5_max = -1e+10;
f_max = -1e+10;
b1 = ceil(size(trainX,1)/2);

for i = 1:m1
    %for r = 1:nR
        for j = 1:m2
            for k = 1:m3
                %for l = 1:m4
                    for m = 1:m5
                        option1.N=2^G2(j);
                        option1.C=2^(G3(k));
                        option1.scale=1;
                        %option1.Scalemode=3;
                        %option1.bias=0;
                        %option1.link=0;
                        option1.L = G1(i);
                        option1.activation = 3;
                        option1.renormal = 1;
                        option1.normal_type = 1;
                        %[~,~,~,~,~,~,f,~,~] = OL_MRVFL(trainX,trainY,testX,testY,option1,nclass,G4(l),G5(m));
                        [~,~,~,~,~,~,f,~,~] = OL_MRVFL(trainX,trainY,testX,testY,option1,nclass,b1,G5(m));
                        if f>=f_max
                            f_max = f;
                            x1_max = G1(i);
                            x2_max = G2(j);
                            x3_max = G3(k);
                            %x4_max = G4(l);
                            x5_max = G5(m);
                        end
                    end
                %end
            end
        end
        
        %update grid
        %int_length_10 = int_length_10/2;
        %int_length_6 = int_length_6/2;
        
        %G2 = linspace(x2_max-int_length_10/2,x2_max+int_length_10/2,m2);
        %G3 = linspace(x3_max-int_length_6/2,x3_max+int_length_6/2,m3);
    %end
end

%Check grid params result
%Options for model
option1.N=2^x2_max;
option1.C=2^(x3_max);
option1.scale=1;
%option1.Scalemode=3;
%option1.bias=0;
%option1.link=0;
option1.L = x1_max;
option1.activation = 3;
option1.renormal = 1;
option1.normal_type = 1;
[model,train_acc,test_acc,~,~,test1,test2,~,~] = OL_MRVFL(trainX,trainY,testX,testY,option1,nclass,b1,x5_max);
%}

%%%REG TEST
%
load airfoilnoisemat.mat
data_all = table2array(airfoilselfnoise);
% load concrete.mat
% data_all = table2array(ConcreteData);
% load dailydemand.mat
% data_all = table2array(DailyDemandForecastingOrders);
% load stock.mat
% data_all = table2array(stockportfolioperformancedatasetS4);
% load wine_red.mat
% data_all = table2array(winequalityred);
% load wine_white.mat
% data_all = table2array(winequalitywhite);
% [row,column]=size(data_all);
% colmin = min(data_all);
% colmax = max(data_all);
% data_all=rescale(data_all,'InputMin',colmin,'InputMax',colmax);
% colmin = min(data_all(:,1:column-1));
% colmax = max(data_all(:,1:column-1));
% data_all=[rescale(data_all(:,1:column-1),'InputMin',colmin,'InputMax',colmax) data_all(:,column)];

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
[trainX,trainY]=normal(trainX,trainY);
[testX,testY]=normal(testX,testY);
nclass = numel(trainY);

c = cvpartition(train_index,'KFold',5);

for i=1:5
    cvtrainx{i} = trainX(training(c,i),:);
    cvtrainy{i} = trainY(training(c,i),:);
    cvtestx{i} = trainX(test(c,i),:);
    cvtesty{i} = trainY(test(c,i),:);
end

%}

%%%NOX
%{
load nox.mat
[trainX,trainY]=normal(train(:,1:end-1),train(:,end));
[testX,testY]=normal(tes(:,1:end-1),tes(:,end));
% trainX = rescale(train(:,1:end-1),'InputMin',min(train(:,1:end-1)),'InputMax',max(train(:,1:end-1)));
% trainY = rescale(train(:,end),'InputMin',min(train(:,end)),'InputMax',max(train(:,end)));
% testX = rescale(tes(:,1:end-1),'InputMin',min(tes(:,1:end-1)),'InputMax',max(tes(:,1:end-1)));
% testY = rescale(tes(:,end),'InputMin',min(tes(:,end)),'InputMax',max(tes(:,end)));
nclass = numel(trainY);

c = cvpartition(size(trainX,1),'KFold',5);

for i=1:5
    cvtrainx{i} = trainX(training(c,i),:);
    cvtrainy{i} = trainY(training(c,i),:);
    cvtestx{i} = trainX(test(c,i),:);
    cvtesty{i} = trainY(test(c,i),:);
end
%}

%GRID SEARCH REG

% settings
%m1 is L, m2 is N, m3 is n for C, m4 is b1size, m5 is olbsize
m1 = 9; m2 = 3; m3 = 10;
%m4 = 6; %For small datasets
%m4 = 5; %For large datasets
m5 = 6;
nR = 5; % number of grid refinememnts
int_length_10 = 10; %interval lengths
int_length_12 = 12;
int_length_6 = 6;
int_length_30 = 30;
int_length_50 = 50;
int_length_100 = 100;
int_length_1000 = 1000;
%construct initial grid
G1 = linspace(2,int_length_10,m1);
G2 = linspace(8,int_length_10,m2);%For small datasets
%G2 = linspace(10,int_length_12,m2);%For large datasets
G3 = linspace(-12,int_length_6,m3);
%G4 = linspace(50,int_length_100,m4);%For small datasets
%G4 = linspace(100,int_length_100,m4);%For large datasets
G5 = linspace(20,int_length_30,m5);%For small datasets
%G5 = linspace(50,int_length_100,m5);%For large datasets

%initial value, must be very small
x1_max = -1e+10;
x2_max = -1e+10;
x3_max = -1e+10;
%x4_max = -1e+10;
x5_max = -1e+10;
if option1.type == 0
    f_max = -1e+10;
elseif option1.type == 1
    rmse_test_min = 100;
    rmse_train_min = 100;
end
pointer_i = cell(m1,1);
pointer_j = cell(m2,1);
pointer_k = cell(m3,1);
pointer_m = cell(m5,1);
pointer_cv = cell(5,1);
for i = 1:m1
    %for r = 1:nR
    for j = 1:m2
        for k = 1:m3
            %for l = 1:m4
            for m = 1:m5
                for cv = 1:5
                    option1.N=2^G2(j);
                    option1.C=2^(G3(k));
                    option1.scale=1;
                    %option1.Scalemode=3;
                    %option1.bias=0;
                    %option1.link=0;
                    option1.L = G1(i);
                    option1.activation = 3;
                    option1.renormal = 1;
                    option1.normal_type = 1;
                    b1 = floor(size(cvtrainx{cv},1)/2);
                    if option1.type == 0
                        [~,~,~,~,~,~,f,~,~] = OL_MRVFL(cvtrainx{cv},cvtrainy{cv},cvtestx{cv},cvtesty{cv},option1,nclass,b1,G5(m));
                        if f>=f_max
                            f_max = f;
                            x1_max = G1(i);
                            x2_max = G2(j);
                            x3_max = G3(k);
                            %x4_max = G4(l);
                            x5_max = G5(m);
                        end
                    elseif option1.type == 1
                        [~,~,~,~,~,~,~,prob1,prob2] = OL_MRVFL(cvtrainx{cv},cvtrainy{cv},cvtestx{cv},cvtesty{cv},option1,nclass,b1,G5(m));
                        %                             pointer2(k,m) = cell2mat(prob1);
                        %pred1 = 0;
                        %pred2 = 0;
                        pred1 = horzcat(prob1{:});
                        
                        %                             rms_temp_train = min(bsxfun(@minus, pred1, trainY),[],2);
                        rms_temp_train = median(pred1,2);
                        %                             for a = 1:G1(i)
                        RMSE_train = sqrt(sum((bsxfun(@minus, rms_temp_train, cvtrainy{cv}).^2),'all')/(size(cvtrainy{cv},1)));
                        %                                 pred1 = pred1 + prob1{a,1};
                        %                                 rms_temp_train{a} = prob1{a,1}-trainY;
                        %                             end
                        %                             rms_temp_train = horzcat(prob1{:});
                        %                             yo_yo = bsxfun(@minus, rms_temp_train, trainY);
                        %                             pred1 = pred1/G1(i);
                        %                             rms_temp_train = pred1-trainY;
                        %                             yo_yo = str2double(rms_temp_train);
                        %                             RMSE_train = sqrt(sum((rms_temp_train.^2),'all')/(size(trainY,1)));
                        %                             RMSE_train = sqrt(sum(((min(yo_yo)).^2),'all')/(size(trainY,1)));
                        pred2 = horzcat(prob2{:});
                        %                             rms_temp_test = min(bsxfun(@minus, pred2, testY),[],2);
                        rms_temp_test = median(pred2,2);
                        %                             for a = 1:G1(i)
                        RMSE_test = sqrt(sum((bsxfun(@minus, rms_temp_test, cvtesty{cv}).^2),'all')/(size(cvtesty{cv},1)));
                        pointer_cv{cv} = RMSE_test;
                        %                                 pred2 = pred2 + prob2{a,1};
                        %                                 rms_temp_test{a} = prob2{a,1}-testY;
                        %                             end
                        %pred2 = pred2/G1(i);
                        %rms_temp_test = pred2-testY;
                        %                             RMSE_test = sqrt(sum((min(rms_temp_test).^2),'all')/(size(testY,1)));
                        if RMSE_test<=rmse_test_min
                            rmse_test_min = RMSE_test;
                            rmse_train_min = RMSE_train;
                            x1_max = G1(i);
                            x2_max = G2(j);
                            x3_max = G3(k);
                            %x4_max = G4(l);
                            x5_max = G5(m);
                        end
                    end
                end
                pointer_m{m} = pointer_cv;
            end
            %end
            pointer_k{k} = pointer_m;
        end
        pointer_j{j} = pointer_k;
    end
    
    %update grid
    %int_length_10 = int_length_10/2;
    %int_length_6 = int_length_6/2;
    
    %G2 = linspace(x2_max-int_length_10/2,x2_max+int_length_10/2,m2);
    %G3 = linspace(x3_max-int_length_6/2,x3_max+int_length_6/2,m3);
    %end
    pointer_i{i} = pointer_j;
end

%Check grid params result

%Options for model
option1.N=2^x2_max;
option1.C=2^(x3_max);
option1.scale=1;
%option1.Scalemode=3;
%option1.bias=0;
%option1.link=0;
option1.L = x1_max;
option1.activation = 3;
option1.renormal = 1;
option1.normal_type = 1;
% pred1 = 0;
% pred2 = 0;
b1 = floor(size(trainX,1)/2);
[model,train_acc,test_acc,~,~,test1,test2,prob1,prob2] = OL_MRVFL(trainX,trainY,testX,testY,option1,nclass,b1,x5_max);

%{
for a = 1:x1_max
   pred1 = pred1 + prob1{a,1};
end
pred1 = pred1/x1_max;
%}
temp_pred1 = cat(2,prob1{:});
pred1 = median(temp_pred1,2);
rms_temp_train = pred1-trainY;
RMSE_train = sqrt(sum((rms_temp_train.^2),'all')/(size(trainY,1)));

%{
for a = 1:x1_max
   pred2 = pred2 + prob2{a,1};
end
pred2 = pred2/x1_max;
%}
temp_pred2 = cat(2,prob2{:});
pred2 = median(temp_pred2,2);
rms_temp_test = pred2-testY;
RMSE_test = sqrt(sum((rms_temp_test.^2),'all')/(size(testY,1)));

%{
rms_temp_train = prob1{1,1}-trainY;
RMSE_train = sqrt(sum((rms_temp_train.^2),'all')/(size(trainY,1)));
rms_temp_test = prob2{1,1}-testY;
RMSE_test = sqrt(sum((rms_temp_test.^2),'all')/(size(testY,1)));
%}
%}
%% ACC for layer 1 to L
train_acc;
test_acc;
test1;
test2;

%Shiva Test over




