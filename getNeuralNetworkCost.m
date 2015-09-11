function [J] = getNeuralNetworkCost(Y, neural_network_output, weights_of_layer, lambda, m)
	regularization_term = (lambda / 2) * get_sum_of_training_weights(weights_of_layer);
	error_in_each_output = (Y .* log(neural_network_output) + (1-Y) .* log(1 - neural_network_output));
	J = (1 / m) * (-1 * sum(sum(error_in_each_output)) + regularization_term);
end

function weight_sum = get_sum_of_training_weights(weights_of_layer)
	weight_sum = 0;
	for i=1:size(weights_of_layer, 2)
		temp = weights_of_layer{i}(2:end,:);
		weight_sum += sum(sum(temp .^ 2));
	endfor
endfunction
