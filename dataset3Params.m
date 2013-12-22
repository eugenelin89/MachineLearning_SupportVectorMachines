function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.0;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


c_array = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_array = c_array;



minError = 1;

counter = 1
for i = 1:size(c_array,2),
  for j = 1:size(sigma_array, 2),
    fprintf('%d. Training with c=%f sigma=%f \n',counter, c_array(i), sigma_array(j));
    model = svmTrain(X,y,c_array(i), @(x1,x2)gaussianKernel(x1,x2,sigma_array(j)));   
    pred = svmPredict(model, Xval);
    error = mean(double(pred~=yval));
    fprintf('Error: %f\n', error);

    if error <= minError,
       minError = error;
       C = c_array(i);
       sigma = sigma_array(j);
    end;

    counter = counter + 1;
  end;
end;

fprintf('Result: C=%f, sigma = %f, error =%f',C,sigma, minError);




% =========================================================================

end
