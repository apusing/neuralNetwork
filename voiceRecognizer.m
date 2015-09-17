function voiceRecognizer(X, y)
	training_set_index = getTrainingSetIndex(X); 

	training_set = X(1:training_set_index,:);
	y_training = y(1:training_set_index);

	disp('Training set size');
	disp(size(training_set));
	disp(size(y_training));
	
	test_set = X(training_set_index+1:end,:);
	y_test = y(training_set_index+1:end);

	disp('Test set size')
	disp(size(test_set));
	disp(size(y_test));

	[final_training_weights] = trainNeuralNetwork(training_set, y_training,...
									0.1, [100], 4);
	save('final_training_weights.mat', 'final_training_weights');

	initiate_testing();
end

function value = getTrainingSetIndex(X)
	trainingSetPerc = 0.8;
	testSetPerc = 0.2;
	examples = size(X, 1);
	value = floor(examples * trainingSetPerc);
	fprintf('Training set index %f\n', value);
end


function initiate_testing(test_set, y_test, final_training_weights)
	final_training_weights = extract_weights_from_unrolled_array([100],...
		final_training_weights);
	activation_values = generate_activation_values(test_set, final_training_weights);
	
end