function activation_of_layer = generate_activation_values(X, weights_of_layer)
	global number_of_hidden_layers;
	
	activation_of_layer = {};
	activation_of_layer{1} = X;
	for layer=2:number_of_hidden_layers+2
		activation_of_layer{layer} = sigmoid((weights_of_layer{layer-1})' *... 
			addRowOfOnes(activation_of_layer{layer-1}));
	endfor

	% printMatrixSize(activation_of_layer, 'activation');
endfunction