function [J gradient] = neuralNetworkCostFunction(training_weights, X, ...
													 y, lambda, hidden_layers)

	% For the purpose of this implementation we are taking the bias node out of the number of nodes 
	% present in the hidden layer array and adding the bias node ourselves
	global num_layers;
	
	Y = generateOutputMatrix(y);

	
	weights_of_layer = extract_weights_from_unrolled_array(hidden_layers,...
								training_weights);

	activation_of_layer = generate_activation_values(X, weights_of_layer);
	
	J = getNeuralNetworkCost(Y, activation_of_layer{end}, ...
		weights_of_layer, lambda);

	gradient = [];
	gradient_values = getGradientMatrix(activation_of_layer, weights_of_layer,...
	 									Y, lambda);
	for i=1:num_layers-1
		gradient = [gradient; (gradient_values{i})(:)];
	endfor
end
