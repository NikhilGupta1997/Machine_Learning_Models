function ans = kernel(x,y)
    tmp = norm(x-y);
    tmp = tmp*tmp;
    ans = exp(-2.5*tmp);
end

fileID= fopen('trainlabels.txt','r');
formatSpec ='%f';
sizeX= [1 inf];
[Y,countY] = fscanf(fileID,formatSpec,sizeX);
Y = Y';
fclose(fileID);
%fprint('dsd');   
M = 2 ;
M = csvread('traindata.txt');
M = M';
[n,m] = size(M);
for i=1:m
    if Y(i) == 2
        Y(i) = -1 ;
    end
end 
Q = zeros(m,m);
for i=1:m
    for j=1:m
        Q(i,j) = -(1/2)*Y(i)*Y(j)*(M(:,i)'*M(:,j));
    end 
end
b = ones(m,1);
C = 500;
cvx_begin
    variable alp(m);
    maximize (quad_form(alp, Q) + b' * alp);
    subject to
        alp >= 0;
        alp - C <= 0;
        alp' * Y == 0;
cvx_end
W = zeros(n,1);
for j = 1:m
   W = W + Y(j)*(alp(j)*M(:,j));
end
b1 = -10000000000;
b2 = 10000000000;
for i = 1:m
    if alp(i) < C - 0.1 & alp(i) > 0.1
        tmp = W'*M(:,i);
        if Y(i) == 1 & tmp < b2
            b2 = tmp;
        elseif Y(i) == -1 & tmp > b1
            b1 = tmp;
        end
    end
end
b = -1/2*(b1+b2);
% Testing
fileID= fopen('testlabels.txt','r');

formatSpec ='%f';
sizeX= [1 inf];
[Y_test,countY_test] = fscanf(fileID,formatSpec,sizeX);
Y_test = Y_test';

fclose(fileID);
%fprint('dsd');   
M1 = csvread('testdata.txt');
%M1 = csvread('traindata.txt');
M1 = M1';
[n1,m1] = size(M1);
for i=1:m1
    if Y_test(i) == 2
        Y_test(i) = -1 ;
    end
end 
ans = zeros(countY_test,1);
for i = 1:m1
    ans(i) = W'*M1(:,i) + b;
    if(ans(i) > 0) 
        ans(i) = 1;
    else
        ans(i) = -1;
    end
end
total = 0;
correct = 0;
for i = 1:m1
    if ans(i) == Y_test(i)
        correct = correct + 1;
    end
    total = total + 1;
end
fprintf(' Accuracy-- Linear Kernel %f ',correct * 100 / total); 

% Code for part c
Q_k = zeros(m,m);
for i=1:m
    for j=1:m
        Q_k(i,j) = -(1/2)*Y(i)*Y(j)*kernel(M(:,i),M(:,j));
    end 
end
b_k = ones(m,1);
C = 500;
cvx_begin
    variable alp_k(m);
    maximize (quad_form(alp_k, Q_k) + b_k' * alp_k);
    subject to
        alp_k >= 0;
        alp_k - C <= 0;
        alp_k' * Y == 0;
cvx_end
b1_k = -10000000000;
b2_k = 10000000000;
for i = 1:m
    if alp_k(i) < C - 0.1 & alp_k(i) > 0.1
        tmp = 0;
        for j = 1:m
            tmp = tmp + alp_k(j)*Y(j)*kernel(M(:,i),M(:,j));
        end    
        if Y(i) == 1 & tmp < b2_k
            b2_k = tmp;
        elseif Y(i) == -1 & tmp > b1_k
            b1_k = tmp;
        end
    end
end
b_k = -1/2*(b1_k+b2_k);

ans_k = zeros(countY_test,1);
for i = 1:m1
    tmp = b_k;
    for j = 1:m
       tmp = tmp + alp_k(j)*Y(j)*kernel(M1(:,i),M(:,j));
    end
    ans_k(i) = tmp;
    if(ans_k(i) > 0) 
        ans_k(i) = 1;
    else
        ans_k(i) = -1;
    end
end
total_k = 0;
correct_k = 0;
for i = 1:m1
    if ans_k(i) == Y_test(i)
        correct_k = correct_k + 1;
    end
    total_k = total_k + 1;
end
fprintf(' Accuracy-- Gaussian Kernel %f ',correct_k * 100 / total_k); 
% % svm part
% %Linear Kernel
%  model = svmtrain(Y, M','-t 0 -c 500');
%  [predicted_label, accuracy, decision_values] = svmpredict(Y_test, M1', model);
%  lin_acc = accuracy(1,1);
% % Gaussian Kernel
% model = svmtrain(Y, M','-g 2.5 -c 500'); 
% [predicted_label, accuracy, decision_values] = svmpredict(Y_test, M1', model);
% gaus_ac = accuracy(1,1);
% C = [1,10,100,1000,10000,100000,1000000];
% accuracies = zeros(7,1);
% test_accuracies = zeros(7,1);
% for i = 1:7    
%     accuracies(i) = svmtrain(Y, M',['-g 2.5 -v 10 -c ',num2str(C(i))]);
%     model = svmtrain(Y, M',['-g 2.5 -c ',num2str(C(i))]); 
%     [predicted_label, accuracy, decision_values] = svmpredict(Y_test, M1', model);
%     test_accuracies(i) = accuracy(1,1);
% end
% figure();
% title('Cross Validation');
% hold on;
%  i = 1:7;
%  plot(i,accuracies,'-r');
%  plot(i,test_accuracies,'-b');
% hold off; 

