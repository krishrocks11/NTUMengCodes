function [test_accuracy,prob_scores] = RVFL_predict(X,Y,model,option)

beta = model.beta;
W = model.W;
b = model.b;
activation = option.activation;

Nsample = size(X,1);

X1 = X*W+repmat(b,Nsample,1);

if activation == 1
    X1 = selu(X1);
elseif activation == 2
    X1 = relu(X1);
elseif activation == 3
    X1 = sigmoid(X1);
elseif activation == 4
    X1 = sin(X1);
elseif activation == 5
    X1 = hardlim(X1);        
elseif activation == 6
    X1 = tribas(X1);
elseif activation == 7
    X1 = radbas(X1);
elseif activation == 8
    X1 = sign(X1);
elseif activation == 9
    X1 = swish(X1);
end

X1=[X1,ones(Nsample,1)];
X = [X,X1];
rawScore = X*beta;

%softmax to generate probabilites for CLASS
if option.type == 0
    rawScore_temp1 = bsxfun(@minus,rawScore,max(rawScore,[],2));
    num = exp(rawScore_temp1);
    dem = sum(num,2);
    prob_scores = bsxfun(@rdivide,num,dem);
elseif option.type == 1 %For REG
    prob_scores = rawScore;
end

[max_prob,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(Y,[],2);
test_accuracy = mean(indx == ind_corrClass);

end
%EOF
