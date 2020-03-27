% Author: Andy Jeong
% Last Updated: November 27, 2019

clear all; close all; clc;
%% Read from csv
data_orig1 = readtable('../1-joints_100frames_human.csv');
confidence1 = make_conf_label(data_orig1);
data1 = table2array((data_orig1(:,1:3)));

data_orig2 = readtable('../1-joints_100frames_mannequin.csv');
confidence2 = make_conf_label(data_orig2);
data2 = table2array((data_orig2(:,1:3)));
%% Prepare video streams
figure(1); figure(2);
vidfile1 = VideoWriter('human.mp4','MPEG-4');
vidfile2 = VideoWriter('mannequin.mp4','MPEG-4');
open(vidfile1); open(vidfile2);

joints_verylow1 = {};
joints_low1 = {};
joints_verylow2 = {};
joints_low2 = {};
%% Scatter plot each body at each frame for all 32 joints
for i = 1:length(data1)/32
    joints = data1((i-1)*32+1:i*32,:);
    conf_level1 = confidence1((i-1)*32+1:i*32,:);
    x = joints(:,1);
    y = joints(:,2);
    z = joints(:,3);
    
    joints2 = data2((i-1)*32+1:i*32,:);
    conf_level2 = confidence2((i-1)*32+1:i*32,:);
    x2 = joints2(:,1);
    y2 = joints2(:,2);
    z2 = joints2(:,3);
    
    % plot human
    figure(1); scatter3(x,y,z ,'k'); hold on;
    % connect skeletal lines
    [ordered_points1, conf_level1] = order_joints(x,y,z, conf_level1);
    [idx_unsure1_1, idx_unsure1_2] = connect_joints(ordered_points1, conf_level1);
    joints_not_seen1 = map_joints(idx_unsure1_1);
    joints_predicted1 = map_joints(idx_unsure1_2);
        
    xlim([min(data1(:,1)), max(data1(:,1))]);
    ylim([min(data1(:,2)), max(data1(:,2))]);
    zlim([min(data1(:,3)), max(data1(:,3))]);
    title(sprintf('%s \n Not seen: %s / Predicted: %s', '<Human>', sprintf('%d,', idx_unsure1_1), sprintf('%d,',idx_unsure1_2)));
    xlabel('x'); ylabel('y'); zlabel('z');
    frames1(i) = getframe(gcf);
    writeVideo(vidfile1,frames1(i));
    
    % plot mannequin
    figure(2); scatter3(x2,y2,z2 ,'k'); hold on;
    % connect skeletal lines
    [ordered_points2, conf_level2] = order_joints(x2,y2,z2, conf_level2);
    [idx_unsure2_1, idx_unsure2_2] = connect_joints(ordered_points2, conf_level2);
    joints_not_seen2 = map_joints(idx_unsure2_1);
    joints_predicted2 = map_joints(idx_unsure2_2);

    xlim([min(data2(:,1)), max(data2(:,1))]);
    ylim([min(data2(:,2)), max(data2(:,2))]);
    zlim([min(data2(:,3)), max(data2(:,3))]);
    title(sprintf('%s \n Not seen: %s / Predicted: %s', '<Mannequin>', sprintf('%d, ', idx_unsure2_1), sprintf('%d,',idx_unsure2_2)));
    xlabel('x'); ylabel('y'); zlabel('z'); 
    frames2(i) = getframe(gcf);
    writeVideo(vidfile2,frames2(i));
    
    % accumulate joint segments of low confidence
    joints_verylow1(end+1:end+length(joints_not_seen1),1) = joints_not_seen1;
    joints_low1(end+1:end+length(joints_predicted1),1) = joints_predicted1;
    joints_verylow2(end+1:end+length(joints_not_seen2),1) = joints_not_seen2;
    joints_low2(end+1:end+length(joints_predicted2),1) = joints_predicted2;
end

figure(1); frame1 = getframe(gcf);
figure(2); frame2 = getframe(gcf);
writeVideo(vidfile1,frame1);
writeVideo(vidfile2,frame2);

close(vidfile1);
close(vidfile2);

% filter
joints_verylow1 = unique(cellstr(joints_verylow1));
joints_low1 = unique(cellstr(joints_low1));
joints_verylow2 = unique(cellstr(joints_verylow2));
joints_low2 = unique(cellstr(joints_low2));

% adjust the size for tabulating
max_size = max([length(joints_verylow1), length(joints_low1), length(joints_verylow2), length(joints_low2)]);
t = cell(max_size); t(:,5:end) = [];
t(1:length(joints_verylow1)+1, 1) = [cellstr("Out of Range - Human");joints_verylow1];
t(1:length(joints_low1)+1, 2) = [cellstr("Predicted - Human"); joints_low1];
t(1:length(joints_verylow2)+1, 3) = [cellstr("Out of Range - Mannequin"); joints_verylow2];
t(1:length(joints_low2)+1, 4) = [cellstr("Predicted - Mannequin"); joints_low2];

% record segments of low confidence
t_table = table(t);
writetable(t_table, 'unsure_joints.csv','WriteVariableNames',false,'Delimiter',',');
