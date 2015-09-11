function [gradient] = getGradientMatrix(activation_of_layer, weights_of_layer,...
										 Y, num_layers, m, lambda)
	delta = {};
	gradient = {};

	delta{num_layers} = activation_of_layer{num_layers} - Y;
	for i=num_layers-1:-1:2
		delta{i} = (activation_of_layer{i} .* (1 - activation_of_layer{i}));
		delta{i} = delta{i} .* (weights_of_layer{i}(2:end,:) * delta{i+1});
	endfor

	% printMatrixSize(delta, 'delta');

	for i=num_layers-1:-1:1
		gradient{i} = addRowOfOnes(activation_of_layer{i}) * (delta{i+1})';
		gradient{i}(2:end,:) = gradient{i}(2:end,:) + lambda * weights_of_layer{i}(2:end,:);
		gradient{i} = gradient{i} / m;
	endfor

	% printMatrixSize(gradient, 'gradient');
end