function [RVFLModel,TrainAcc,TestAcc,prob1,prob2]  = RVFL(trainX,trainY,testX,testY,option,b1_size,olb_size)

% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

% Train RVFL
%[RVFLModel,TrainAcc] = RVFL_train(trainX,trainY,option);
[RVFLModel,TrainAcc,prob1] = OL_RVFL_train(trainX,trainY,option,b1_size,olb_size);

% Using trained model, predict the testing data
[TestAcc,prob2] = RVFL_predict(testX,testY,RVFLModel,option);

end
%EOF