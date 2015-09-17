function weights_of_layer = extract_weights_from_unrolled_array(hidden_layers,...
										training_weights)
	global number_of_features;
	global number_of_hidden_layers;
	global num_labels;
	weights_of_layer = {};

	temp = hidden_layers(1)*(number_of_features+1);
	weights_of_layer{1} = reshape_call(training_weights,...
							 temp, 1, number_of_features + 1);

	for layer=2:number_of_hidden_layers
		temp2 = (hidden_layers(layer-1) + 1) * hidden_layers(layer); % one added for bias node
		weights_of_layer{layer} = reshape_call(training_weights, temp2, temp+1, ...
			hidden_layers(layer-1)+1);
		temp = temp + temp2;
	endfor

	temp2 = (hidden_layers(end)+1) * num_labels;
	weights_of_layer{number_of_hidden_layers+1} = reshape_call(training_weights, temp2,...
	 temp+1, hidden_layers(end)+1);

	% printMatrixSize(weights_of_layer, 'weight');
end

function result = reshape_call(weights, num_elements, start_index, row_size)
	result = reshape(weights(start_index:start_index+num_elements-1),... 
			row_size, num_elements/row_size);
endfunction

