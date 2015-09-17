function printMatrixSize(M, type_of_values)
	fprintf('Size of matrix %d\n', size(M,2))
	for i=1:size(M,2)
		fprintf('The %s values for layer %d is: %d X %d\n', type_of_values, i, size(M{i},1), size(M{i},2));
	endfor
end