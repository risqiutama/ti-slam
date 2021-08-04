%% Graph optimization, by taking into account uncertainty from MDN
% Note that without outlier rejection, the output can be very noisy.

% Copy the following list odometry sequences and loop files either for in-house robot or handheld
% experiments

% In-house ground robot
list_odom_seq = {'32', '33', '34', ...
                 '37', '39', '42', ...
                 '43', '44', '45', ...
                 '46', '47', '48', ...
                 '49'};
list_loop_files = {'2019-10-24-18-22-33', '2019-11-23-15-54-25', '2019-11-23-15-52-53', ...
                '2019-11-23-15-59-12', '2019-11-04-20-29-51', '2019-11-22-10-10-00', ...
                '2019-11-22-10-14-01', '2019-11-22-10-22-48', '2019-11-22-10-26-42', ...
                '2019-11-22-10-34-57', '2019-11-22-10-37-42', '2019-11-22-10-38-47', ...
                '2019-11-28-15-40-10'};

% Handheld
list_odom_seq = {'35', '36', '37', ...
                 '38', '39', '40', ...
                 '42', '43'};
list_loop_files = {'2020-01-28-11-28-50', '2020-01-28-11-34-05', '2020-01-28-11-39-07', ...
                   '2020-01-28-11-55-54', '2020-01-28-12-01-24', '2020-02-06-15-35-57', ...
                   '2020-02-07-15-57-43', '2020-02-07-17-01-24'};

%%

clc, clear, close all

% Setting
odom_name = 'neural_odometry';
embedding_name = 'neural_embedding';
loop_pose_name = 'neural_loop_closure';
loop_thres = '0.045';
spatial_consistency_thres = 0.7; % should be between 0.01 to 1
weight_odom = 0.01;
weight_loop = 300; % should be bigger than weight_odom, try increasing/decreasing the zero

% =====

% Replace both the list of odom seq and list loop files with the respective
% dataset
list_odom_seq = {'32', '33', '34', ...
                 '37', '39', '42', ...
                 '43', '44', '45', ...
                 '46', '47', '48', ...
                 '49'};
list_loop_files = {'2019-10-24-18-22-33', '2019-11-23-15-54-25', '2019-11-23-15-52-53', ...
                '2019-11-23-15-59-12', '2019-11-04-20-29-51', '2019-11-22-10-10-00', ...
                '2019-11-22-10-14-01', '2019-11-22-10-22-48', '2019-11-22-10-26-42', ...
                '2019-11-22-10-34-57', '2019-11-22-10-37-42', '2019-11-22-10-38-47', ...
                '2019-11-28-15-40-10'};

format long;

main_dir = 'Python/';
working_dir = strcat(main_dir, 'odometry/results/');

output_folder = strcat('figures/optimized_odometry/', odom_name, '_', loop_pose_name);
if ~exist(output_folder, 'dir')
       mkdir(output_folder)
end

base_odom_filename = strcat(odom_name, '_epbest_seq');
base_sigma_odom_filename = strcat('sigmapose_', odom_name, '_epbest_seq');

base_loop_filename = strcat('pose_', loop_pose_name, '_', embedding_name, '_epbest_');
gt_filename = 'gt_seq';
informationmatrix = [1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1];

for j = 1:size(list_odom_seq,1)
    disp(strcat('Processing Sequences: ', list_odom_seq{j}, ' - ', list_loop_files{j}))
    
    % Load odometry
    odom_pathname = strcat(working_dir, base_odom_filename, list_odom_seq{j}, '.txt');
    odom_array = csvread(odom_pathname);
    
    sigma_pathname = strcat(working_dir, base_sigma_odom_filename, list_odom_seq{j}, '.txt');
    sigma_array = csvread(sigma_pathname);
    
    % === Construct the pose graph based on odometry constraints ===
    pg3D = poseGraph3D;
    len = size(odom_array,1);
    
    last_row_mat = [0 0 0 1];
    for k = 2:len
        temp_traj_k_1 = vec2mat(odom_array(k-1,:), 4);
        temp_traj_k_1 = [temp_traj_k_1; last_row_mat];
        temp_traj_k = vec2mat(odom_array(k,:), 4);
        temp_traj_k = [temp_traj_k; last_row_mat];
        relativePose = temp_traj_k_1\temp_traj_k;
        % Relative orientation represented in quaternions
        relativeQuat = tform2quat(relativePose);
        % Relative pose as [x y z qw qx qy qz] 
        relativePose = [tform2trvec(relativePose),relativeQuat];
        
        % Extract information matrix
        covariance = [sigma_array(k,1) 0 0 0 0 0; 
                      0 sigma_array(k,2) 0 0 0 0;
                      0 0 sigma_array(k,3) 0 0 0;
                      0 0 0 sigma_array(k,4) 0 0;
                      0 0 0 0 sigma_array(k,5) 0;
                      0 0 0 0 0 sigma_array(k,6)];
        inv_cov = inv(weight_odom * covariance);
        informationmatrix = [inv_cov(1) 0 0 0 0 0 inv_cov(8) ...
                            0 0 0 0 inv_cov(15) 0 0 0 inv_cov(22) ...
                            0 0 inv_cov(29) 0 inv_cov(36)];
        
        % Add pose to pose graph
        addRelativePose(pg3D,relativePose,informationmatrix, k-1, k);
    end
    
    % === Load loop closure constraint, and add to the graph ===
    loop_pathname = strcat(working_dir, base_loop_filename, list_loop_files{j}, ...
                    '_', loop_thres, '.csv');
    loop_pairs_array = readtable(loop_pathname);
    loop_len = size(loop_pairs_array, 1);
    
    % Spatial Consistency Check
    consistent_loop_pairs = spatial_consistency(loop_pairs_array, odom_array, spatial_consistency_thres);

    for i = 1:size(consistent_loop_pairs, 1)
        % Convert euler angle (degree) to quaternion
        eul = [double(deg2rad(consistent_loop_pairs(i,6))), double(deg2rad(consistent_loop_pairs(i,7))), ...
            double(deg2rad(consistent_loop_pairs(i,8)))];
        quat_pose = eul2quat(eul);
        relativePose = [double(consistent_loop_pairs(i,3)), double(consistent_loop_pairs(i,4)), ...
            double(consistent_loop_pairs(i,5)), ...
            quat_pose(1), quat_pose(2), quat_pose(3), quat_pose(4)];
        
        % Extract information matrix
        covariance = [consistent_loop_pairs(i,9) 0 0 0 0 0; 
                      0 consistent_loop_pairs(i,10) 0 0 0 0;
                      0 0 consistent_loop_pairs(i,11) 0 0 0;
                      0 0 0 consistent_loop_pairs(i,12) 0 0;
                      0 0 0 0 consistent_loop_pairs(i,13) 0;
                      0 0 0 0 0 consistent_loop_pairs(i,14)];
        inv_cov = inv(weight_loop * covariance);
        informationmatrix = [inv_cov(1) 0 0 0 0 0 inv_cov(8) ...
                            0 0 0 0 inv_cov(15) 0 0 0 inv_cov(22) ...
                            0 0 inv_cov(29) 0 inv_cov(36)];

        addRelativePose(pg3D,relativePose,informationmatrix,...
                    consistent_loop_pairs(i,1),consistent_loop_pairs(i,2));       
    end
    
    % === Optimize pose graph ===
    optimizedPosegraph = optimizePoseGraph(pg3D, "g2o-levenberg-marquardt");
    optimizedposes = nodes(optimizedPosegraph);
    
    % === Plotting ===
    g = figure;
    axis equal
    
    % Load ground truth for plotting
    gt_pathname = strcat(working_dir, gt_filename, list_odom_seq{j}, '.txt');
    gt_array = csvread(gt_pathname);
    
    gt_array_origin_x = gt_array(1:end,4)-gt_array(1,4);
    gt_array_origin_y = gt_array(1:end,8)-gt_array(1,8);
    gt_array_origin_z = gt_array(1:end,12)-gt_array(1,12);
    
    plot(gt_array_origin_x, gt_array_origin_y, ....
        '--', 'color', [0.2500    0.2500    0.2500], 'LineWidth', 3);
    hold on
    plot(odom_array(:,4),odom_array(:,8), 'color', [0.3010    0.7450    0.9330], 'LineWidth', 3);
    hold on
    plot(optimizedposes(:,1),optimizedposes(:,2), 'color', [0.6350    0.0780    0.1840], 'LineWidth', 3);
    
    legend('Ground truth', 'Odometry', 'Optimized Trajectory');

    output_fullpath = strcat(output_folder, '/optimized_traj_seq', list_odom_seq{j}, ...
                        '_',list_loop_files{j},'.pdf');
    set(g,'Units','Inches');
    pos = get(g,'Position');
    set(g,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(g,output_fullpath,'-dpdf','-r0', '-fillpage')
    
    ate_slam = sqrt(((optimizedposes(:, 1) - gt_array_origin_x).^2 + ...
                        (optimizedposes(:, 2) - gt_array_origin_y).^2 + ...
                    (optimizedposes(:, 3) - gt_array_origin_z).^2)/3);
    rmse_ate_odom = mean(sqrt(((odom_array(:, 4) - gt_array_origin_x).^2 + ...
                        (odom_array(:, 8) - gt_array_origin_y).^2 + ...
                    (odom_array(:, 12) - gt_array_origin_z).^2)/3));
    improvement = ((rmse_ate_odom - mean(ate_slam)) / rmse_ate_odom)*100;   
    disp('RMSE ATE TI-SLAM (m):')
    disp(mean(ate_slam));
    disp('Variance ATE TI-SLAM (m):')
    disp(var(ate_slam));
    disp('RMSE odometry (m):')
    disp(rmse_ate_odom);
    disp('Improvement (%):')
    disp(improvement);
end