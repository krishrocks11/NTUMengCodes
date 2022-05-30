function [model,train_accuracy,prob_scores] = OL_RVFL_train(trainX,trainY,option,b1_size,olb_size)


N = option.N;
C = option.C;
s = 1;
activation = option.activation;

[Nsample,Nfea] = size(trainX);

%making batches
%num_of_batches = 6;
initial_batch_size = b1_size;%Size of the first batch size
batch_size = olb_size;%Nsample/num_of_batches;%OL batch size
k = 1;
first_batch_trainX = trainX(1:initial_batch_size,:);
first_batch_trainY = trainY(1:initial_batch_size,:);
for nn = initial_batch_size+1:batch_size:(Nsample - rem( Nsample-initial_batch_size , batch_size ) ) 
    
   batch_trainX{k} = trainX(nn:nn+batch_size-1,:);
   batch_trainY{k} = trainY(nn:nn+batch_size-1,:);
   k = k+1;
end
if (rem( Nsample-initial_batch_size , batch_size ) ~= 0)
    batch_trainX{k} = trainX((Nsample - rem( Nsample-initial_batch_size , batch_size ) )+1:Nsample,:);
    batch_trainY{k} = trainY((Nsample - rem( Nsample-initial_batch_size , batch_size ) )+1:Nsample,:);
end

A_input = first_batch_trainX;%batch_trainX{1};

W = (rand(Nfea,N)*2*s-1);
b = s*rand(1,N);
X1 = A_input*W+repmat(b,initial_batch_size,1);

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


X = [A_input,X1]; 


X = [X,ones(initial_batch_size,1)];%bias in the output layer

if size(X,2)<initial_batch_size
    beta = (eye(size(X,2))/C+X'*X) \ X'*first_batch_trainY;
else
    beta = X'*((eye(size(X,1))/C+X*X') \ first_batch_trainY);
end

%For OL
K = (eye(size(X,2))/C+X'*X);

for j = 1:k-1
    A_input = batch_trainX{j};
    
    X1 = A_input*W+repmat(b,batch_size,1);

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


    X = [A_input,X1]; 


    X = [X,ones(batch_size,1)];%bias in the output layer
    Block=size(X,1);
    K=K-K*X'*((eye(Block)+X*K*X')\X*K);
    beta=beta+K*X'*(batch_trainY{j}-X*beta);

end

if (rem( Nsample-initial_batch_size , batch_size ) ~= 0)
    remain_batch_size = rem( Nsample-initial_batch_size , batch_size );
    A_input = batch_trainX{k};
    %OL
    X1 = A_input*W+repmat(b,remain_batch_size,1);

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


    X = [A_input,X1]; 

    %Shiva - insert online sequential component here
    X = [X,ones(remain_batch_size,1)];%bias in the output layer
    Block=size(X,1);
    K=K-K*X'*((eye(Block)+X*K*X')\X*K);
    beta=beta+K*X'*(batch_trainY{k}-X*beta);
end

model.beta = beta; %output weights
model.W = W; %input-hidden layer weights
model.b = b; %hidden layer bias




%For ACC
X1 = trainX*W+repmat(b,Nsample,1);
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


X = [trainX,X1]; 


X = [X,ones(Nsample,1)];%bias in the output layer

trainY_temp = X*beta; %output of RVFL

%softmax to generate probabilites for CLASS
if option.type == 0
    trainY_temp1 = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
    num = exp(trainY_temp1);
    dem = sum(num,2);
    prob_scores = bsxfun(@rdivide,num,dem);
elseif option.type == 1%FOR REG
    prob_scores = trainY_temp;
end
[max_prob,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(trainY,[],2);
train_accuracy = mean(indx == ind_corrClass);

end
%EOF