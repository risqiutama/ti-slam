%% Function to check spatial consistency among loop closure constraints
%
% 
function consistent_loop_pairs = spatial_consistency(loop_pairs_array, odom_array, threshold)
    last_row_mat = [0 0 0 1];
    loop_len = size(loop_pairs_array, 1);
    
    is_first_found = 0; % check whether you got one
    for i = 1:loop_len
        pair_i = [loop_pairs_array.Var1(i), loop_pairs_array.Var3(i)];
        max_j = i+(loop_len);
        if max_j > loop_len
            max_j = loop_len;
        end
        limit = i + 50;
        if limit > max_j
            limit = max_j;
        end
        for j = (i+1):limit
            p1_rot_matrix = [odom_array(pair_i(2), 1) odom_array(pair_i(2), 2) odom_array(pair_i(2), 3);
                             odom_array(pair_i(2), 5) odom_array(pair_i(2), 6) odom_array(pair_i(2), 7);
                             odom_array(pair_i(2), 9) odom_array(pair_i(2), 10) odom_array(pair_i(2), 11)];
            abs_p1_rot = rotm2eul(p1_rot_matrix, "XYZ");
            abs_p1_trans = [odom_array(pair_i(2), 4), odom_array(pair_i(2), 8), ...
                            odom_array(pair_i(2), 12)];

            pair_j = [loop_pairs_array.Var1(j), loop_pairs_array.Var3(j)];
            
            if pair_j(1) <= size(odom_array,1) && pair_j(2) <= size(odom_array,1) && ...
                    pair_i(1) <= size(odom_array,1) && pair_i(2) <= size(odom_array,1)
                abs_p2 = vec2mat(odom_array(pair_j(2), :), 4);
                abs_p2 = [abs_p2; last_row_mat];

                % Get relative pose 1
                eul_j = [double(deg2rad(loop_pairs_array.Var10(j))), double(deg2rad(loop_pairs_array.Var9(j))), ...
                        double(deg2rad(loop_pairs_array.Var8(j)))];
                rotm_j = eul2rotm(eul_j);
                tform_j = rotm2tform(rotm_j);
                tform_j(13) = double(loop_pairs_array.Var5(j));
                tform_j(14) = double(loop_pairs_array.Var6(j));
                tform_j(15) = double(loop_pairs_array.Var7(j));
                rel_pose_1 = inv(tform_j);

                % Get relative pose 2
                abs_p3 = vec2mat(odom_array(pair_j(1), :), 4);
                abs_p3 = [abs_p3; last_row_mat];

                abs_p4 = vec2mat(odom_array(pair_i(1), :), 4);
                abs_p4 = [abs_p4; last_row_mat];

                rel_pose_2 = abs_p3 \ abs_p4;

                % Get relative pose 3
                eul_i = [double(deg2rad(loop_pairs_array.Var10(i))), double(deg2rad(loop_pairs_array.Var9(i))), ...
                        double(deg2rad(loop_pairs_array.Var8(i)))];
                rotm_i = eul2rotm(eul_i);
                tform_i = rotm2tform(rotm_i);
                tform_i(13) = double(loop_pairs_array.Var5(i));
                tform_i(14) = double(loop_pairs_array.Var6(i));
                tform_i(15) = double(loop_pairs_array.Var7(i));
                rel_pose_3 = tform_i;

                final_cyclic_pose = abs_p2 * rel_pose_1;
                final_cyclic_pose = final_cyclic_pose * rel_pose_2;
                final_cyclic_pose = final_cyclic_pose * rel_pose_3;
                final_cyclic_pose_eul = rotm2eul(final_cyclic_pose(1:3,1:3), "XYZ");

                diff_pose = sqrt(((abs_p1_trans(1) - final_cyclic_pose(13)).^2 + ...
                                (abs_p1_trans(2) - final_cyclic_pose(14)).^2 + ...
                                (abs_p1_trans(3) - final_cyclic_pose(15)).^2 + ...
                                (abs_p1_rot(1) - final_cyclic_pose_eul(1)).^2 + ...
                                (abs_p1_rot(2) - final_cyclic_pose_eul(2)).^2 + ...
                                (abs_p1_rot(3) - final_cyclic_pose_eul(3)).^2)/6);

                if diff_pose < threshold % 0.005
                    disp(strcat('Found robust loop!', num2str(diff_pose)))
                    if is_first_found > 0
                        pairs_i = [loop_pairs_array.Var1(i), loop_pairs_array.Var3(i), ...
                            loop_pairs_array.Var5(i), loop_pairs_array.Var6(i), ...
                            loop_pairs_array.Var7(i), loop_pairs_array.Var8(i), ...
                            loop_pairs_array.Var9(i), loop_pairs_array.Var10(i), ...
                            loop_pairs_array.Var11(i), loop_pairs_array.Var12(i), ...
                            loop_pairs_array.Var13(i), loop_pairs_array.Var14(i), ...
                            loop_pairs_array.Var15(i), loop_pairs_array.Var16(i)];
                        consistent_loop_pairs = [consistent_loop_pairs; pairs_i];
                        pairs_j = [loop_pairs_array.Var1(j), loop_pairs_array.Var3(j), ...
                            loop_pairs_array.Var5(j), loop_pairs_array.Var6(j), ...
                            loop_pairs_array.Var7(j), loop_pairs_array.Var8(j), ...
                            loop_pairs_array.Var9(j), loop_pairs_array.Var10(j), ...
                            loop_pairs_array.Var11(i), loop_pairs_array.Var12(i), ...
                            loop_pairs_array.Var13(i), loop_pairs_array.Var14(i), ...
                            loop_pairs_array.Var15(i), loop_pairs_array.Var16(i)];
                        consistent_loop_pairs = [consistent_loop_pairs; pairs_j];
                    else 
                            pairs_i = [loop_pairs_array.Var1(i), loop_pairs_array.Var3(i), ...
                            loop_pairs_array.Var5(i), loop_pairs_array.Var6(i), ...
                            loop_pairs_array.Var7(i), loop_pairs_array.Var8(i), ...
                            loop_pairs_array.Var9(i), loop_pairs_array.Var10(i), ...
                            loop_pairs_array.Var11(i), loop_pairs_array.Var12(i), ...
                            loop_pairs_array.Var13(i), loop_pairs_array.Var14(i), ...
                            loop_pairs_array.Var15(i), loop_pairs_array.Var16(i)];
                        consistent_loop_pairs = [pairs_i];
                        pairs_j = [loop_pairs_array.Var1(j), loop_pairs_array.Var3(j), ...
                            loop_pairs_array.Var5(j), loop_pairs_array.Var6(j), ...
                            loop_pairs_array.Var7(j), loop_pairs_array.Var8(j), ...
                            loop_pairs_array.Var9(j), loop_pairs_array.Var10(j), ...
                            loop_pairs_array.Var11(i), loop_pairs_array.Var12(i), ...
                            loop_pairs_array.Var13(i), loop_pairs_array.Var14(i), ...
                            loop_pairs_array.Var15(i), loop_pairs_array.Var16(i)];
                        consistent_loop_pairs = [consistent_loop_pairs; pairs_j];
                            is_first_found = 1;
                    end
                end
            end
        end
    end
end