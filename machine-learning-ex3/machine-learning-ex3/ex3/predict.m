function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%vi co 5000 training example
%xem hinh matrix_training.jpg
temp1 = [ones(m, 1) X];
temp2 = [ones(m, 1) sigmoid(temp1 * Theta1')];
temp3 = sigmoid(temp2 * Theta2');
%temp3 la mot matrix 5000*10, voi moi dong(10) la xac suat cua so o index tuong ung
[maxTemp3, maxTemp3_2] = max(temp3');
%[matrix_value, matrix_ind]
p = maxTemp3_2';






% =========================================================================


end
