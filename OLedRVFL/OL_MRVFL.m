function [Model,TrainAcc,TestAcc,TrainingTime,TestingTime,acc1,acc2,prob1,prob2] = ...
    OL_MRVFL(trainX,trainY,testX,testY,option,num_classes,b1_size,olb_size)

seed = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(seed);

% Train RVFL
%[Model,TrainAcc,TrainingTime,~] = MRVFLtrain(trainX,trainY,option);

% Using trained model, predict the testing data
%[TestAcc,TestingTime,~] = MRVFLpredict(testX,testY,Model,option);

%Shiva Code
%Train
[Model,TrainAcc,TrainingTime,prob1,acc1] = OL_MRVFLtrain(trainX,trainY,option, num_classes,b1_size,olb_size);
%Test
[acc2,TestAcc,TestingTime,prob2] = OL_MRVFLpredict(testX,testY,Model,option,num_classes);

end
%EOF