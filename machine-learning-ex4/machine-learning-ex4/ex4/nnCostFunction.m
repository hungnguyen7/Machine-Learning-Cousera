function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network


%size 25x401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%size 10x26
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); 
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1) X]; %them nut bias cho a1, %matrix 5000x401
a2 = [ones(m,1) sigmoid(a1*Theta1')]; %tinh hidden layer 2 va them nut bias, matrix 5000x26
a3 = sigmoid(a2*Theta2');

%repmat: repeat matrix
%repmat(matrix, so luong hang se repeat matrix, so luong cot se repeat matrix)
%toan tu == so sanh 2 thanh phan cung vi tri giua hai matrix 

repmat1 = repmat([1:num_labels], m, 1); %matrix 5000x10
repmat2 = repmat(y, 1, num_labels); %matrix 5000x10
yVector = repmat1 == repmat2; %so sanh 2 vector de co matrix 5000x10 ma o do, gia su, o vi tri so 2 tuong ung se la so 1, tat ca la so 0

% A.*B: moi phan tu cua A nhan tuong ung voi moi phan tu cua B 
cost = -yVector.*log(a3) - (1 - yVector) .* log(1-a3);

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

%Tong ham J
J = (1/m) * sum(sum(cost)) + (lambda / (2*m))*(sum(sum(Theta1NoBias.^2)) + sum(sum(Theta2NoBias.^2)));


delta_cua_moi_unit_trong_layer2 = zeros(size(Theta1));
delta_cua_moi_unit_trong_layer3 = zeros(size(Theta2));

% BACKPROPAGATION ALGORITHM
for t=1:m, %duyet qua tung training example
    tmp_a1 = a1(t, :)'; %vector 401
    tmp_a2 = a2(t, :)'; %vector 26
    tmp_a3 = a3(t, :)'; %vector 10
    tmp_yVector = yVector(t,:)';
    
    %sai so cua ket qua va thuc te
    sai_so_cua_traning_ex_thu_t_o_output = tmp_a3 - tmp_yVector;
    sai_so_cua_traning_ex_thu_t_o_hidden_layer2 = Theta2'*sai_so_cua_traning_ex_thu_t_o_output .* tmp_a2 .* (1-tmp_a2); %1
    % 1 Tuong duong
    % z2t = [1; Theta1 * tmp_a1];
	  % d2t = Theta2' * sai_so_cua_traning_ex_thu_t_o_output .* sigmoidGradient(z2t);
    %Tong dao ham tung phan theo theta
    
    delta_cua_moi_unit_trong_layer2 = delta_cua_moi_unit_trong_layer2 + sai_so_cua_traning_ex_thu_t_o_hidden_layer2(2:end)*tmp_a1'; %tu 2->end vi bo nut bias
    delta_cua_moi_unit_trong_layer3 = delta_cua_moi_unit_trong_layer3 + sai_so_cua_traning_ex_thu_t_o_output*tmp_a2';
    %DOC KI: moi lan vong lap for se traing 1 ex, sau so thuat toan BACKPROPAGATION ALGORITHM se tinh ra cac delta sau khi da training,
    %vi training m ex nen can cong cac gia tri delta tuong ung lai voi nhau thi moi co du sai so khi training tat ca m training ex
endfor;

Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1NoBias ];
Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2NoBias ];

Theta1_grad = (1/m)*delta_cua_moi_unit_trong_layer2 + (lambda/m) * Theta1ZeroBias;
Theta2_grad = (1/m)*delta_cua_moi_unit_trong_layer3 + (lambda/m) * Theta2ZeroBias;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
