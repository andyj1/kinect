function [ordered_joints, conf_level] = order_joints(x,y,z, confidence)
    num_joints = 32;
    % ignore pelvis, which doesn't have a parent joint
    % number of joints * 3d space (x,y,z) * two joints
    ordered_joints = zeros([num_joints-1, 3, 2]); 
    ordered_joints(:,:,1) = [x(2:end,:), y(2:end,:), z(2:end,:)];  
    % 1. SPINE_NAVAL - PELVIS
    ordered_joints(1,:,2) = [x(1), y(1), z(1)];
    % 2. SPINE_CHEST - SPINE_NAVAL
    ordered_joints(2,:,2) = [x(2), y(2), z(2)];
    % 3. NECK - SPINE_CHEST
    ordered_joints(3,:,2) = [x(3), y(3), z(3)];
    % 4. CLAVICLE_LEFT - SPINE_CHEST
    ordered_joints(4,:,2) = [x(3), y(3), z(3)];
    % 5. SHOULDER_LEFT - CLAVICLE_LEFT
    ordered_joints(5,:,2) = [x(5), y(5), z(5)];
    % 6. ELBOW_LEFT - SHOULDER_LEFT
    ordered_joints(6,:,2) = [x(6), y(6), z(6)];
    % 7. WRIST_LEFT - ELBOW_LEFT
    ordered_joints(7,:,2) = [x(7), y(7), z(7)];
    % 8. HAND_LEFT - WRIST_LEFT
    ordered_joints(8,:,2) = [x(8), y(8), z(8)];
    % 9. HANDTIP_LEFT - HAND_LEFT
    ordered_joints(9,:,2) = [x(9), y(9), z(9)];
    %10. THUMB_LEFT - WRIST_LEFT
    ordered_joints(10,:,2) = [x(8), y(8), z(8)];
    %11. CLAVICLE_RIGHT - SPINE_CHEST
    ordered_joints(11,:,2) = [x(3), y(3), z(3)];
    %12. SHOULDER_RIGHT - CLAVICLE_RIGHT
    ordered_joints(12,:,2) = [x(12), y(12), z(12)];
    %13. ELBOW_RIGHT - SHOULDER_RIGHT
    ordered_joints(13,:,2) = [x(13), y(13), z(13)];
    %14. WRIST_RIGHT - ELBOW_RIGHT
    ordered_joints(14,:,2) = [x(14), y(14), z(14)];
    %15. HAND_RIGHT - WRIST_RIGHT
    ordered_joints(15,:,2) = [x(15), y(15), z(15)];
    %16. HANDTIP_RIGHT - HAND_RIGHT
    ordered_joints(16,:,2) = [x(16), y(16), z(16)];
    %17. THUMB_RIGHT - WRIST_RIGHT
    ordered_joints(17,:,2) = [x(15), y(15), z(15)];
    %18. HIP_LEFT - PELVIS
    ordered_joints(18,:,2) = [x(1), y(1), z(1)];
    %19. KNEE_LEFT - HIP_LEFT
    ordered_joints(19,:,2) = [x(19), y(19), z(19)];
    %20. ANKLE_LEFT - KNEE_LEFT
    ordered_joints(20,:,2) = [x(20), y(20), z(20)];
    %21. FOOT_LEFT - ANKLE_LEFT
    ordered_joints(21,:,2) = [x(21), y(21), z(21)];
    %22. HIP_RIGHT - PELVIS
    ordered_joints(22,:,2) = [x(1), y(1), z(1)];
    %23. KNEE_RIGHT - HIP_RIGHT
    ordered_joints(23,:,2) = [x(23), y(23), z(23)];
    %24. ANKLE_RIGHT - KNEE_RIGHT
    ordered_joints(24,:,2) = [x(24), y(24), z(24)];
    %25. FOOT_RIGHT - ANKLE_RIGHT
    ordered_joints(25,:,2) = [x(25), y(25), z(25)];
    %26. HEAD - NECK
    ordered_joints(26,:,2) = [x(4), y(4), z(4)];
    %27. NOSE - HEAD
    ordered_joints(27,:,2) = [x(27), y(27), z(27)];
    %28. EYE_LEFT - HEAD
    ordered_joints(28,:,2) = [x(27), y(27), z(27)];
    %29. EAR_LEFT - HEAD
    ordered_joints(29,:,2) = [x(27), y(27), z(27)];
    %30. EYE_RIGHT - HEAD
    ordered_joints(30,:,2) = [x(27), y(27), z(27)];
    %31. EAR_RIGHT - HEAD
    ordered_joints(31,:,2) = [x(27), y(27), z(27)];
       
    % compare confidence level for segments
    conf_level = zeros([num_joints-1, 2]); 
    conf_level(:,1) = confidence(2:end,:);  
    % 1. SPINE_NAVAL - PELVIS
    conf_level(1,2) = confidence(1);
    % 2. SPINE_CHEST - SPINE_NAVAL
    conf_level(2,2) = confidence(2);
    % 3. NECK - SPINE_CHEST
    conf_level(3,2) = confidence(3);
    % 4. CLAVICLE_LEFT - SPINE_CHEST
    conf_level(4,2) = confidence(3);
    % 5. SHOULDER_LEFT - CLAVICLE_LEFT
    conf_level(5,2) = confidence(5);
    % 6. ELBOW_LEFT - SHOULDER_LEFT
    conf_level(6,2) = confidence(6);
    % 7. WRIST_LEFT - ELBOW_LEFT
    conf_level(7,2) = confidence(7);
    % 8. HAND_LEFT - WRIST_LEFT
    conf_level(8,2) = confidence(8);
    % 9. HANDTIP_LEFT - HAND_LEFT
    conf_level(9,2) = confidence(9);
    %10. THUMB_LEFT - WRIST_LEFT
    conf_level(10,2) = confidence(8);
    %11. CLAVICLE_RIGHT - SPINE_CHEST
    conf_level(11,2) = confidence(3);
    %12. SHOULDER_RIGHT - CLAVICLE_RIGHT
    conf_level(12,2) = confidence(12);
    %13. ELBOW_RIGHT - SHOULDER_RIGHT
    conf_level(13,2) = confidence(13);
    %14. WRIST_RIGHT - ELBOW_RIGHT
    conf_level(14,2) = confidence(14);
    %15. HAND_RIGHT - WRIST_RIGHT
    conf_level(15,2) = confidence(15);
    %16. HANDTIP_RIGHT - HAND_RIGHT
    conf_level(16,2) = confidence(16);
    %17. THUMB_RIGHT - WRIST_RIGHT
    conf_level(17,2) = confidence(15);
    %18. HIP_LEFT - PELVIS
    conf_level(18,2) = confidence(1);
    %19. KNEE_LEFT - HIP_LEFT
    conf_level(19,2) = confidence(19);
    %20. ANKLE_LEFT - KNEE_LEFT
    conf_level(20,2) = confidence(20);
    %21. FOOT_LEFT - ANKLE_LEFT
    conf_level(21,2) = confidence(21);
    %22. HIP_RIGHT - PELVIS
    conf_level(22,2) = confidence(1);
    %23. KNEE_RIGHT - HIP_RIGHT
    conf_level(23,2) = confidence(23);
    %24. ANKLE_RIGHT - KNEE_RIGHT
    conf_level(24,2) = confidence(24);
    %25. FOOT_RIGHT - ANKLE_RIGHT
    conf_level(25,2) = confidence(25);
    %26. HEAD - NECK
    conf_level(26,2) = confidence(4);
    %27. NOSE - HEAD
    conf_level(27,2) = confidence(27);
    %28. EYE_LEFT - HEAD
    conf_level(28,2) = confidence(27);
    %29. EAR_LEFT - HEAD
    conf_level(29,2) = confidence(27);
    %30. EYE_RIGHT - HEAD
    conf_level(30,2) = confidence(27);
    %31. EAR_RIGHT - HEAD
    conf_level(31,2) = confidence(27);
end



