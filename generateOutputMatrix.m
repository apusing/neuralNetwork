function [Y] = generateOutputMatrix(y)
	global num_labels;
	global m;
	Y = zeros(num_labels, m);

	for i = 1:m
		Y(y(i), i) = 1;
	endfor
end