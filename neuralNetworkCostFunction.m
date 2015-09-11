function [J gradient] = neuralNetworkCostFunction(training_weights, num_labels, X, ...
													 y, lambda, hidden_layers)

	% For the purpose of this implementation we are taking the bias node out of the number of nodes 
	% present in the hidden layer array and adding the bias node ourselves
	Y = generateOutputMatrix(y, num_labels);
	global number_of_features number_of_hidden_layers num_layers;
	global m activation_of_layer weights_of_layer delta;
	
	number_of_features = size(X,1);
	number_of_hidden_layers = size(hidden_layers,2);
	num_layers = number_of_hidden_layers + 2;
	m = size(y,1);
	activation_of_layer = {};
	weights_of_layer = {};
	delta = {};

	extract_weights_from_unrolled_array(hidden_layers, training_weights, num_labels);

	generate_activation_values(X);
	
	J = getNeuralNetworkCost(Y, activation_of_layer{end}, ...
		weights_of_layer, lambda, m);

	gradient = [];
	gradient_values = getGradientMatrix(activation_of_layer, weights_of_layer,...
	 									Y, num_layers, m, lambda);
	for i=1:num_layers-1
		gradient = [gradient; (gradient_values{i})'(:)];
	endfor
end


function result = reshape_call(weights, num_elements, start_index, row_size)
	result = reshape(weights(start_index:start_index+num_elements-1),... 
			row_size, num_elements/row_size);
endfunction


function extract_weights_from_unrolled_array(hidden_layers, training_weights, num_labels)
	global number_of_features;
	global number_of_hidden_layers;
	global weights_of_layer;

	temp = hidden_layers(1)*(number_of_features+1);
	weights_of_layer{1} = reshape_call(training_weights, temp, 1, number_of_features + 1);

	for layer=2:number_of_hidden_layers
		temp2 = (hidden_layers(layer-1) + 1) * hidden_layers(layer); % one added for bias node
		weights_of_layer{i} = reshape_call(training_weights, temp2, temp+1, ...
			hidden_layers(layer-1)+1);
		temp = temp + temp2;
	endfor

	temp2 = (hidden_layers(end)+1) * num_labels;
	weights_of_layer{number_of_hidden_layers+1} = reshape_call(training_weights, temp2,...
	 temp+1, hidden_layers(end)+1);

	% printMatrixSize(weights_of_layer, 'weight');
endfunction

function generate_activation_values(X)
	global activation_of_layer;
	global weights_of_layer;
	global number_of_hidden_layers;

	activation_of_layer{1} = X;
	for layer=2:number_of_hidden_layers+2
		activation_of_layer{layer} = sigmoid((weights_of_layer{layer-1})' *... 
			addRowOfOnes(activation_of_layer{layer-1}));
	endfor

	% printMatrixSize(activation_of_layer, 'activation');
endfunction