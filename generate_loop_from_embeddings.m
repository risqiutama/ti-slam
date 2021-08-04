%% Generate loop pairs from the embedding features

% Copy the following list files either for in-house robot or handheld
% experiments

% In-house ground robot
list_files = {'2019-10-24-18-22-33', '2019-11-23-15-54-25', '2019-11-23-15-52-53', ...
                '2019-11-23-15-59-12', '2019-11-04-20-29-51', '2019-11-22-10-10-00', ...
                '2019-11-22-10-14-01', '2019-11-22-10-22-48', '2019-11-22-10-26-42', ...
                '2019-11-22-10-34-57', '2019-11-22-10-37-42', '2019-11-22-10-38-47', ...
                '2019-11-28-15-40-10'};

% Handheld
list_files = {'2020-01-28-11-28-50', '2020-01-28-11-34-05', '2020-01-28-11-39-07', ...
              '2020-01-28-11-55-54', '2020-01-28-12-01-24', '2020-02-06-15-35-57', ...
              '2020-02-07-15-57-43', '2020-02-07-17-01-24'};

%% 

clc, clear, close all

embed_threshold = 0.045;
model_name = 'neural_embedding';
data_folder = 'Python/loop/results/';
output_folder = 'Python/odometry/results/';
if ~exist(output_folder, 'dir')
       mkdir(output_folder)
end

% Replace test files based on the dataset you used
list_files = {'2019-10-24-18-22-33', '2019-11-23-15-54-25', '2019-11-23-15-52-53', ...
                '2019-11-23-15-59-12', '2019-11-04-20-29-51', '2019-11-22-10-10-00', ...
                '2019-11-22-10-14-01', '2019-11-22-10-22-48', '2019-11-22-10-26-42', ...
                '2019-11-22-10-34-57', '2019-11-22-10-37-42', '2019-11-22-10-38-47', ...
                '2019-11-28-15-40-10'};

base_filename = strcat('embedding_', model_name, '_epbest_');
tril_val = -18;

for j = 1:size(list_files,2)
    embedding_file = strcat(data_folder, base_filename, list_files{j}, '.csv');
    embedding_array = csvread(embedding_file);
    embed_distance = pdist(embedding_array, 'cosine');
    embed_square = squareform(embed_distance);
    embed_tril = tril(embed_square,tril_val);
    embed_filtered = (embed_tril < embed_threshold & embed_tril > 0);
    [row_loop, col_loop] = find(embed_filtered);
    output_file = strcat(output_folder, model_name, '_', list_files{j}, ...
        '_', string(embed_threshold), '.csv');
    disp(strcat('Generating : ', output_file));
    writematrix([row_loop, col_loop],output_file,'Delimiter',',') 
end