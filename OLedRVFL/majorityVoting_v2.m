%Function by Shiva. Incomplete
function acc = majorityVoting_v2(Y,probs,j,input_size,num_classes,L)
%For now inout size is fixed
%input_size = 150;
test_rows = input_size;
nclasses = num_classes;
% Get classes from prob scores
for i=1:j
    [MaxProb,indx] = max(probs{i},[],2); 
    class{i} = indx;
end
                                           
%[MaxProb,indx] = max(AvgProbScores,[],2);
%[~, Ind_corrClass] = max(Y,[],2);
%acc = mean(indx == Ind_corrClass);

% Concatenate all prediciton arrays into one big matrix. 
compositepreds = class{1};
for k=2:length(class)
    compositepreds = cat(2,compositepreds,class{i});%compositepreds + class{k};
end
%compositepreds = [class{1}, class{2}];%, class{3}, class{4}, class{5}, class{6}, class{7}, class{8}, class{9}, class{10}];

%majvote with mode

indx_majvot = mode (compositepreds,2);
[~, Ind_corrClass] = max(Y,[],2);
acc = mean(indx_majvot == Ind_corrClass);


%Test maj vote v2
%{
Final_decision = zeros(test_rows,1);%test_rows=input size
all_results = [1:nclasses]; %possible outcomes = nclasses
for row = 1:test_rows
    election_array = zeros(1,nclasses); %what is the significance of the params here?
    for col = 1:j %for different classifiers l
       election_array(compositepreds(row,col)) = ... 
           election_array(compositepreds(row,col)) + 1;
    end 
    [~,I] = max(election_array);
    Final_decision(row) = all_results(I);
end
acc_2 = mean(Final_decision == Ind_corrClass);
%}
end