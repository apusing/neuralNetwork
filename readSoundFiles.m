function [X,y] = readSoundFiles()
	global directory_name = '/home/apusing/Documents/personal/Coursera/Machine_Learning/sound_dir/audio_books/wav/';
	files = readdir(directory_name);
	X = [];
	y = [];
	for i=1:size(files,1)
		file_name = files{i};
		if strmatch('.', file_name) == 1
			disp('ignoring file');
		else
			[temp_X temp_y] = generateInputForFile(file_name);
			X = [X; temp_X];
			y = [y; temp_y];
		end
	endfor
	[X y] = randomizeData(X, y)
	save('audio_input.mat', 'X');
	save('audio_output.mat', 'y');
end


function [X y] = randomizeData(X, y)
	n = rand(size(X,1), 1);
	[garbage index] = sort(n);
	X = X(index,:);
	y = y(index);
end


function [X, y] = generateInputForFile(file_name)
	global directory_name;
	disp(strcat(directory_name, '/', file_name));
	[input_,fs,nbits] = wavread(strcat(directory_name, '/', file_name));
	cols = 8000;
	rows = floor(size(input_, 1)/cols);
	X = reshape(input_(1:rows*cols), rows, cols);
	y = ones(rows, 1) * getClassValue(file_name);
end


function val = getClassValue(file_name)
	if strcmp(file_name, 'little1.wav') == 1
		val = 1
	elseif strcmp(file_name, 'red1.wav') == 1
		val = 2
	elseif strcmp(file_name, 'robinson1.wav') == 1
		val = 3
	else
		val = 4
	endif
end