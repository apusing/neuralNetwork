function [new_matrix] = addRowOfOnes(X)

new_matrix = [ones(1, size(X, 2)); X];

end