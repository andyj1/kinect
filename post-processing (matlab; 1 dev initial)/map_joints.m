function [segments] = map_joints(indices)
    sprintf("started");
    
%     segments = cell([31,1]);
    segments = {};
    for i = 1:length(indices)
        idx = indices(i,1);
        switch (idx)
            case 1
                segments(i,1) = {'SPINE_NAVAL - PELVIS'};
            case 2
                segments(i,1) = {'SPINE_CHEST - SPINE_NAVAL'};
            case 3
                segments(i,1) = {'NECK - SPINE_CHEST'};
            case 4
                segments(i,1) = {'CLAVICLE_LEFT - SPINE_CHEST'};
            case 5
                segments(i,1) = {'SHOULDER_LEFT - CLAVICLE_LEFT'};
            case 6
                segments(i,1) = {'ELBOW_LEFT - SHOULDER_LEFT'};
            case 7
                segments(i,1) = {'WRIST_LEFT - ELBOW_LEFT'};
            case 8
                segments(i,1) = {'HAND_LEFT - WRIST_LEFT'};
            case 9
                segments(i,1) = {'HANDTIP_LEFT - HAND_LEFT'};
            case 10
                segments(i,1) = {'THUMB_LEFT - WRIST_LEFT'};
            case 11
                segments(i,1) = {'CLAVICLE_RIGHT - SPINE_CHEST'};
            case 12
                segments(i,1) = {'SHOULDER_RIGHT - CLAVICLE_RIGHT'};
            case 13
                segments(i,1) = {'ELBOW_RIGHT - SHOULDER_RIGHT'};
            case 14
                segments(i,1) = {'WRIST_RIGHT - ELBOW_RIGHT'};
            case 15
                segments(i,1) = {'HAND_RIGHT - WRIST_RIGHT'};
            case 16
                segments(i,1) = {'HANDTIP_RIGHT - HAND_RIGHT'};
            case 17
                segments(i,1) = {'THUMB_RIGHT - WRIST_RIGHT'};
            case 18
                segments(i,1) = {'HIP_LEFT - PELVIS'};
            case 19
                segments(i,1) = {'KNEE_LEFT - HIP_LEFT'};
            case 20
                segments(i,1) = {'ANKLE_LEFT - KNEE_LEFT'};
            case 21
                segments(i,1) = {'FOOT_LEFT - ANKLE_LEFT'};
            case 22
                segments(i,1) = {'HIP_RIGHT - PELVIS'};
            case 23
                segments(i,1) = {'KNEE_RIGHT - HIP_RIGHT'};
            case 24
                segments(i,1) = {'ANKLE_RIGHT - KNEE_RIGHT'};
            case 25
                segments(i,1) = {'FOOT_RIGHT - ANKLE_RIGHT'};
            case 26
                segments(i,1) = {'HEAD - NECK'};
            case 27
                segments(i,1) = {'NOSE - HEAD'};
            case 28
                segments(i,1) = {'EYE_LEFT - HEAD'};
            case 29
                segments(i,1) = {'EAR_LEFT - HEAD'};
            case 30
                segments(i,1) = {'EYE_RIGHT - HEAD'};
            case 31
                segments(i,1) = {'EAR_RIGHT - HEAD'};
        end
    end
end