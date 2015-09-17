function [final_training_weights] = trainNeuralNetwork(X, y, lambda, hiddenLayers, K)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Initialize global varaibles
	global number_of_features number_of_hidden_layers num_layers m num_labels;
	number_of_features = size(X,2);
	number_of_hidden_layers = size(hiddenLayers,2);
	num_layers = number_of_hidden_layers + 2;
	m = size(y,1);
	num_labels = K;
	fprintf('features = %d, hiddenLayers = %d, layers = %d, m = %d, labels = %d\n',...
			number_of_features, number_of_hidden_layers, num_layers, m, num_labels);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% For Weights equal to 1

	weights = getWeightsInitializedToOne(hiddenLayers, X, K);
	initial_training_weights = getUnrolledArray(weights);

	fprintf('\nFeedforward Using Neural Network ...\n');
	lambda = 0;

	J = neuralNetworkCostFunction(initial_training_weights, ...
							X', y, lambda, hiddenLayers);
	grad = [];
	fprintf('Cost at parameters (loaded from ex4weights): %f\n', J);

	fprintf('\nProgram paused. Press enter to continue.\n');
	pause;
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% For randomly initialized weights
	weights = getRandomlyInitializedWeights(hiddenLayers, X, K);
	initial_training_weights = getUnrolledArray(weights);

	fprintf('\nFeedforward Using Neural Network ...\n')
	lambda = 1;

	J = neuralNetworkCostFunction(initial_training_weights, ...
							X', y, lambda, hiddenLayers);
	grad = [];
	fprintf('Cost at parameters (loaded from ex4weights): %f\n', J);

	fprintf('\nProgram paused. Press enter to continue.\n');
	pause;
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Using fmincg to train our neural network
	fprintf('\nTraining Neural Network... \n');
	options = optimset('MaxIter', 50);
	lambda = 1;

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) neuralNetworkCostFunction(p, ...
										X', y, lambda, hiddenLayers);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[final_training_weights cost] = fmincg(costFunction, initial_training_weights,...
										options);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Now extract the weights and save them in a file and run the test on them
	% from a separate file

end

function unrolled = getUnrolledArray(weights)
	unrolled = [];
	for i=1:size(weights,2)
		unrolled = [unrolled; weights{i}(:)];
	end
	fprintf('Unrolled array size %d\n', size(unrolled, 1));
end


function weights = getRandomlyInitializedWeights(hiddenLayers, X, K)
	global number_of_hidden_layers;
	weights = {};

	weights{1} = randInitializeWeights(size(X, 2) + 1, hiddenLayers(1));
	
	for i = 2:number_of_hidden_layers
		weights{i} = randInitializeWeights(hiddenLayers(i-1)+1, hiddenLayers(i));
	endfor

	weights{number_of_hidden_layers + 1} = ...
		randInitializeWeights(hiddenLayers(number_of_hidden_layers)+1, K);

	printMatrixSize(weights, 'weights');
end


function weights = getWeightsInitializedToOne(hiddenLayers, X, K)
	global number_of_hidden_layers;
	weights = {};

	weights{1} = ones(size(X, 2) + 1, hiddenLayers(1));
	
	for i = 2:number_of_hidden_layers
		weights{i} = ones(hiddenLayers(i-1)+1, hiddenLayers(i));
	endfor

	weights{number_of_hidden_layers + 1} = ...
		ones(hiddenLayers(number_of_hidden_layers)+1, K);

	printMatrixSize(weights, 'weights');
end