function [idx_unsure_1, idx_unsure_2] = connect_joints(ordered_joints, confidence)
    % concatenate child - parent x,y,z coordinates
    x = [ordered_joints(:,1,1), ordered_joints(:,1,2)];
    y = [ordered_joints(:,2,1), ordered_joints(:,2,2)];
    z = [ordered_joints(:,3,1), ordered_joints(:,3,2)];
    conf_level = [0; min(confidence,[],2)];
    num_joints = 32;
    idx_unsure_1 = []; idx_unsure_2 = [];
    i_1 = 1; i_2 = 1;
    % connect the two joints into a segment
    for j = 1:num_joints-2
        tmp_x = [x(j,:), x(j+1,:)];
        tmp_y = [y(j,:), y(j+1,:)];
        tmp_z = [z(j,:), z(j+1,:)];
        if conf_level(j+1) == 3
            plot3(tmp_x,tmp_y,tmp_z,'b'); hold on;
        else
            if conf_level(j+1) == 2
                % append unsure points
                idx_unsure_2(i_2,1) = j;
                i_2 = i_2 + 1;
                % plot in 3D
                plot3(tmp_x,tmp_y,tmp_z,'g--'); hold on;
            elseif conf_level(j+1) == 1
                % append unsure points
                idx_unsure_1(i_1,1) = j;
                i_1 = i_1 + 1;
                % plot in 3D
                plot3(tmp_x,tmp_y,tmp_z,'r--'); hold on;
            end
        end
    end
    hold off;   
end