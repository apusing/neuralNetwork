function [J grad] = trainNeuralNetwork(X, y, lambda, hiddenLayers, K)

% Create a weights array, using randomized values,
% theta unrolled into just one array
% It will contain all the weights required to train our neural network

% network. We then call our cost and gradient function and check if cost
% and gradient are being computed correctly. This function will be used to 

% We then train our Neural Network using an existing gradient descent implementation
% fmincg and output the theta values in a output file named training_weights

% Once this is done you can now run neural network and get the prediction values

% Create "short hand" for the cost function to be minimized
initial_training_weights;
costFunction = @(training_weights) neuralNetworkCostFunction(training_weights, ...
                                   K, X, y, lambda, hiddenLayers);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_training_weights, options);
