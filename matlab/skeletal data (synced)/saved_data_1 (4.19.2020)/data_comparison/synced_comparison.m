clc;clear all;close all;

%% read in the data
single = readtable('./joints_single_master.csv');
double = readtable('./joints_sync_synced.csv');
triple = readtable('./joints_gen_sync.csv');

single_pos = single(:,3:5);
double_pos = double(:,3:5);
triple_pos = triple(:,3:5);

single_pos = table2array(single_pos);
double_pos = table2array(double_pos);
triple_pos = table2array(triple_pos);
%% plot each frame of 32 joints
joints_cell = num2cell(1:32);
joints_cell = {'1','2','3','4','5','6','7','8','9','10',...
                '11','12','13','14','15','16','17','18','19','20',...
                '21','22','23','24','25','26','27','28','29','30',...
                '31','32'};

figure('Renderer', 'painters', 'Position', [50 50 900 600]);
for frame = 1:1:992/32
    joints = 1:length(joints_cell);
    
    % x
    subplot(311);
    plot(joints, single_pos(frame:frame+31,1)');
    hold on;
    plot(joints, double_pos(frame:frame+31,1)');
    plot(joints, triple_pos(frame:frame+31,1)');
    legend('1 device','2 devices','3 devices','Location','Best');
    title('x');
    xlim([1 32]);
    xlabel('joint label (1-32)');
    ylabel('position (mm)');
    hold off;
    
    % y
    subplot(312);
    plot(joints, single_pos(frame:frame+31,2)');
    hold on;
    plot(joints, double_pos(frame:frame+31,2)');
    plot(joints, triple_pos(frame:frame+31,2)');
    legend('1 device','2 devices','3 devices','Location','Best');
    title('y');
    xlim([1 32]);
    xlabel('joint label (1-32)');
    ylabel('position (mm)');
    hold off;
    
    % z
    subplot(313);
    plot(joints, single_pos(frame:frame+31,3)');
    hold on;
    plot(joints, double_pos(frame:frame+31,3)');
    plot(joints, triple_pos(frame:frame+31,3)');
    legend('1 device','2 devices','3 devices','Location','Best');
    title('z');
    xlim([1 32]);
    xlabel('joint label (1-32)');
    ylabel('position (mm)');
    hold off;
    
end

saveas(gcf,'x,y,z_positions.png');
