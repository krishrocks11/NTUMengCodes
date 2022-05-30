function [model,TrainingAccuracy,Training_time,ProbScores, majvotacc] = OL_MRVFLtrain(trainX,trainY,option,num_classes,b1_size,olb_size)

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

N = option.N;
L = option.L;
C = option.C;
activation = option.activation;
s = 1;  %scaling factor


A = cell(L,1); %for L hidden layers
beta = cell(L,1);
weights = cell(L,1);
biases = cell(L,1);
mu = cell(L,1);
sigma = cell(L,1);
TrainingAccuracy = zeros(L,1);
ProbScores = cell(L,1); %depends on number of hidden layer

A_input = first_batch_trainX;%batch_trainX{1};

tic    

%Shiva - first batch input to train weights
for i = 1:L
    
    if i==1
        w = s*2*rand(Nfea,N)-1;
    else
        w = s*2*rand(Nfea+N,N)-1;
    end
    
    b = s*rand(1,N);
    weights{i} = w;
    biases{i} = b;
    
    A1 = A_input * w+repmat(b,initial_batch_size,1); %replaced Nsample by batch_size
    if option.renormal == 1
        if option.normal_type ==0
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
        end
    end  
    if activation == 1
        A1 = selu(A1);
    elseif activation == 2
        A1 = relu(A1);
    elseif activation == 3
        A1 = sigmoid(A1);
    elseif activation == 4
        A1 = sin(A1);
    elseif activation == 5
        A1 = hardlim(A1);        
    elseif activation == 6
        A1 = tribas(A1);
    elseif activation == 7
        A1 = radbas(A1);
    elseif activation == 8
        A1 = sign(A1);
    elseif activation == 9
        A1 = swish(A1);
    end
    %{
    if option.renormal == 1
        if option.normal_type ==1
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;layer normalization
        end
    end
    %}
    
%     A1 = A_input * w;
%     mu{i} = mean(A1,1);
%     sigma{i} = std(A1);
%     A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
%     A1 = A1+repmat(b,Nsample,1);
%     A1 = relu(A1);
    
%     A1 = A_input * w+repmat(b,Nsample,1);
%     A1 = sigmoid(A1);
    
    %A1_temp1 = [A_input,A1,ones(Nsample,1)];
    A1_temp1 = [first_batch_trainX,A1,ones(initial_batch_size,1)]; %replaced Nsample by batch_size
    [M,beta1]  = OL_l2_weights(A1_temp1,first_batch_trainY,C,initial_batch_size); %replaced Nsample by batch_size
    K{i} = inv(M);
    A{i} =  A1_temp1;
    beta{i} = beta1;
    %shiva - insert online sequential component here???
    
    %clear A1 A1_temp1 A1_temp2 beta1
    A_input = [first_batch_trainX A1];
    
    
   %% Calculate the training accuracy
   %{
    trainY_temp = A1_temp1*beta1;
    
    %MajorityVoting
%     [max_score,indx] = max(trainY_temp,[],2);
%     pred_idx(:,i) = indx;

    %%softmax to generate probabilites for classification or direct for regression
        if option.type == 0
            trainY_temp = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
            prob_scores=softmax(trainY_temp')';
            ProbScores{i,1} = prob_scores;

        elseif option.type == 1
            ProbScores{i} = trainY_temp;
        end

    %One layer's accuracy
%     [~,indx] = max(prob_scores,[],2);
%     [~, ind_corrClass] = max(trainY,[],2);
%     correct_index{i} = (indx == ind_corrClass);

    
    %Calculate the training accuracy for first i layers
    TrainingAccuracy(i,1) = ComputeAcc(first_batch_trainY,ProbScores,i); %averaging prob.scores
    %TrainingAccuracy(i,1) = majorityVoting(trainY,pred_idx); %majority voting
    %}
end
%L = L;
w = weights;
b = biases;
%beta = beta;
%mu = mu;
%sigma = sigma;


%Shiva - online learning
for j = 1:k-1
    A_input = batch_trainX{j};
    for i = 1:L
        %{
        if i==1
            w = s*2*rand(Nfea,N)-1;
        else
            w = s*2*rand(Nfea+N,N)-1;
        end
    
        b = s*rand(1,N);
        weights{i} = w;
        biases{i} = b;
    
        A1 = A_input * w+repmat(b,batch_size,1); %replaced Nsample by batch_size
        %}
        A1 = A_input * w{i}+ repmat(b{i},batch_size,1);
        if option.renormal == 1
            if option.normal_type ==0
                mu{i} = mean(A1,1);
                sigma{i} = std(A1);
                A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
            end
        end  
        if activation == 1
            A1 = selu(A1);
        elseif activation == 2
            A1 = relu(A1);
        elseif activation == 3
            A1 = sigmoid(A1);
        elseif activation == 4
            A1 = sin(A1);
        elseif activation == 5
            A1 = hardlim(A1);        
        elseif activation == 6
            A1 = tribas(A1);
        elseif activation == 7
            A1 = radbas(A1);
        elseif activation == 8
            A1 = sign(A1);
        elseif activation == 9
            A1 = swish(A1);
        end
        %{
        if option.renormal == 1
            if option.normal_type ==1
                mu{i} = mean(A1,1);
                sigma{i} = std(A1);
                A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;layer normalization
            end
        end
        %}
    
%        A1 = A_input * w;
%        mu{i} = mean(A1,1);
%        sigma{i} = std(A1);
%        A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
%        A1 = A1+repmat(b,Nsample,1);
%        A1 = relu(A1);
    
%        A1 = A_input * w+repmat(b,Nsample,1);
%        A1 = sigmoid(A1);
    
        %Shiva - insert online sequential component here
        A1_temp1 = [batch_trainX{j},A1,ones(batch_size,1)]; %replaced Nsample by batch_size
        Block=size(A1_temp1,1);
        K{i}=K{i}-K{i}*A1_temp1'*((eye(Block)+A1_temp1*K{i}*A1_temp1')\A1_temp1*K{i});
        beta1=beta{i}+K{i}*A1_temp1'*(batch_trainY{j}-A1_temp1*beta{i});
        %A1_temp1 = [A_input,A1,ones(Nsample,1)];
        %A1_temp1 = [trainX,A1,ones(batch_size,1)]; %replaced Nsample by batch_size
        %beta1  = l2_weights(A1_temp1,trainY,C,batch_size); %replaced Nsample by batch_size
    
        A{i} =  A1_temp1;
        beta{i} = beta1;
        
    
        %clear A1 A1_temp1 A1_temp2 beta1
        A_input = [batch_trainX{j} A1];
    
    
       %% Calculate the training accuracy
       %{
        trainY_temp = A1_temp1*beta1;
        
    
        %MajorityVoting
%        [max_score,indx] = max(trainY_temp,[],2);
%        pred_idx(:,i) = indx;
        
        %%softmax to generate probabilites for classification or direct for regression
        if option.type == 0
            trainY_temp = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
            prob_scores=softmax(trainY_temp')';
            ProbScores{i,1} = prob_scores;

        elseif option.type == 1
            ProbScores{i} = trainY_temp;
        end

        %One layer's accuracy
%        [~,indx] = max(prob_scores,[],2);
%        [~, ind_corrClass] = max(trainY,[],2);
%        correct_index{i} = (indx == ind_corrClass);

    
        %Calculate the training accuracy for first i layers
        TrainingAccuracy(i,1) = ComputeAcc(batch_trainY{j},ProbScores,i); %averaging prob.scores
        %TrainingAccuracy(i,1) = majorityVoting(trainY,pred_idx); %majority voting
        %}
    end
end

if (rem( Nsample-initial_batch_size , batch_size ) ~= 0)
    remain_batch_size = rem( Nsample-initial_batch_size , batch_size );
    A_input = batch_trainX{k};
    for i = 1:L
        %{
        if i==1
            w = s*2*rand(Nfea,N)-1;
        else
            w = s*2*rand(Nfea+N,N)-1;
        end
    
        b = s*rand(1,N);
        weights{i} = w;
        biases{i} = b;
    
        A1 = A_input * w+repmat(b,batch_size,1); %replaced Nsample by batch_size
        %}
        A1 = A_input * w{i}+ repmat(b{i},remain_batch_size,1);
        if option.renormal == 1
            if option.normal_type ==0
                mu{i} = mean(A1,1);
                sigma{i} = std(A1);
                A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
            end
        end  
        if activation == 1
            A1 = selu(A1);
        elseif activation == 2
            A1 = relu(A1);
        elseif activation == 3
            A1 = sigmoid(A1);
        elseif activation == 4
            A1 = sin(A1);
        elseif activation == 5
            A1 = hardlim(A1);        
        elseif activation == 6
            A1 = tribas(A1);
        elseif activation == 7
            A1 = radbas(A1);
        elseif activation == 8
            A1 = sign(A1);
        elseif activation == 9
            A1 = swish(A1);
        end
        %{
        if option.renormal == 1
            if option.normal_type ==1
                mu{i} = mean(A1,1);
                sigma{i} = std(A1);
                A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;layer normalization
            end
        end
        %}
    
%        A1 = A_input * w;
%        mu{i} = mean(A1,1);
%        sigma{i} = std(A1);
%        A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
%        A1 = A1+repmat(b,Nsample,1);
%        A1 = relu(A1);
    
%        A1 = A_input * w+repmat(b,Nsample,1);
%        A1 = sigmoid(A1);
    
        %Shiva - insert online sequential component here
        A1_temp1 = [batch_trainX{k},A1,ones(remain_batch_size,1)]; %replaced Nsample by batch_size
        Block=size(A1_temp1,1);
        K{i}=K{i}-K{i}*A1_temp1'*((eye(Block)+A1_temp1*K{i}*A1_temp1')\A1_temp1*K{i});
        beta1=beta{i}+K{i}*A1_temp1'*(batch_trainY{k}-A1_temp1*beta{i});
        %A1_temp1 = [A_input,A1,ones(Nsample,1)];
        %A1_temp1 = [trainX,A1,ones(remain_batch_size,1)]; %replaced Nsample by batch_size
        %beta1  = l2_weights(A1_temp1,trainY,C,remain_batch_size); %replaced Nsample by batch_size
    
        A{i} =  A1_temp1;
        beta{i} = beta1;
        
    
        %clear A1 A1_temp1 A1_temp2 beta1
        A_input = [batch_trainX{k} A1];
    
    
       %% Calculate the training accuracy
       %{
        trainY_temp = A1_temp1*beta1;
    
        %MajorityVoting
%        [max_score,indx] = max(trainY_temp,[],2);
%        pred_idx(:,i) = indx;
        
        
        %%softmax to generate probabilites for classification or direct for regression
        if option.type == 0
            trainY_temp = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
            prob_scores=softmax(trainY_temp')';
            ProbScores{i,1} = prob_scores;

        elseif option.type == 1
            ProbScores{i} = trainY_temp;
        end

        %One layer's accuracy
%        [~,indx] = max(prob_scores,[],2);
%        [~, ind_corrClass] = max(trainY,[],2);
%        correct_index{i} = (indx == ind_corrClass);

    
        %Calculate the training accuracy for first i layers
        TrainingAccuracy(i,1) = ComputeAcc(batch_trainY{k},ProbScores,i); %averaging prob.scores
        %TrainingAccuracy(i,1) = majorityVoting(trainY,pred_idx); %majority voting
        %}
    end
end

%For maj vote v2
A_input = trainX;
for i = 1:L
    A1 = A_input * w{i}+ repmat(b{i},Nsample,1);
    if option.renormal == 1
        if option.normal_type ==0
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
        end
    end
    if activation == 1
        A1 = selu(A1);
    elseif activation == 2
        A1 = relu(A1);
    elseif activation == 3
        A1 = sigmoid(A1);
    elseif activation == 4
        A1 = sin(A1);
    elseif activation == 5
        A1 = hardlim(A1);        
    elseif activation == 6
        A1 = tribas(A1);
    elseif activation == 7
        A1 = radbas(A1);
    elseif activation == 8
        A1 = sign(A1);
    elseif activation == 9
        A1 = swish(A1);
    end
    %{
    if option.renormal == 1
        if option.normal_type ==1
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
        end
    end
    %}
    
%     A1 = A_input * w{i};
%     A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
%     A1 = A1+ repmat(b{i},Nsample,1);
%     A1 = relu(A1);
    
%     A1 = A_input * w{i}+ repmat(b{i},Nsample,1);
%     A1 = sigmoid(A1);
    
    %A1_temp1 = [A_input,A1,ones(Nsample,1)]; 
    A1_temp1 = [trainX,A1,ones(Nsample,1)]; 
    
    %A1_temp3 = A1; 
    A{i} = A1_temp1;
    %clear A1 A1_temp1 A1_temp2 w1 b1
    A_input = [trainX A1];
    
    %% Calculate the testing accuracy
    beta_temp = beta{i};
    testY_temp = A1_temp1*beta_temp;
    
    %%MajorityVoting
    %[max_score,indx] = max(testY_temp,[],2);
    %pred_idx(:,i) = indx;

    %%softmax to generate probabilites for classification or direct for regression
    if option.type == 0
        testY_temp = bsxfun(@minus,testY_temp,max(testY_temp,[],2)); %for numerical stability
        prob_scores=softmax(testY_temp')';
        ProbScores{i} = prob_scores;
    
    elseif option.type == 1
        ProbScores{i} = testY_temp;
    end
end
%Shiva MajVote
%preds = [];
majvotacc = majorityVoting_v2(trainY,ProbScores,i,Nsample,num_classes,L);
Training_time = toc;


%%

model.L = L;
model.w = weights;
model.b = biases;
model.beta = beta;
model.mu = mu;
model.sigma = sigma;

end