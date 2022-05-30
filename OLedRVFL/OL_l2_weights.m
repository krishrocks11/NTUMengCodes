function [K,x] = OL_l2_weights(A,b,C,Nsample)

if C == 0
    x = pinv(A)*b;
    K = (eye(size(A,2))/C+A'*A);%K = ?;
else
    if size(A,2)<Nsample
        x = (eye(size(A,2))/C+A'*A) \ A'*b;
        K = (eye(size(A,2))/C+A'*A);
    else
        x = A'*((eye(size(A,1))/C+A*A') \ b);
        K = (eye(size(A,2))/C+A'*A);%K = ?;
    end
end


end
% toc

