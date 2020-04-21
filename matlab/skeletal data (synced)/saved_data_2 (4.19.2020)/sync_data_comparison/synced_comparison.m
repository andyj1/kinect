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

singlex = zeros(1,32)';
doublex = zeros(1,32)';
triplex = zeros(1,32)';
singley = zeros(1,32)';
doubley = zeros(1,32)';
tripley = zeros(1,32)';
singlez = zeros(1,32)';
doublez = zeros(1,32)';
triplez = zeros(1,32)';
% accumulate
for frame = 1:1:960/32-1
    frame
    joints = 1:length(joints_cell);
    singlex = singlex + single_pos(frame:frame+31,1); 
    doublex = doublex + double_pos(frame:frame+31,1); 
    triplex = triplex + triple_pos(frame:frame+31,1); 
    
    singley = singley + single_pos(frame:frame+31,2); 
    doubley = doubley + double_pos(frame:frame+31,2); 
    tripley = tripley + triple_pos(frame:frame+31,2); 
    
    singlez = singlez + single_pos(frame:frame+31,3); 
    doublez = doublez + double_pos(frame:frame+31,3); 
    triplez = triplez + triple_pos(frame:frame+31,3); 
end

% divide by count
singlex = singlex./(960/32-1);
doublex = doublex./(960/32-1);
triplex = triplex./(960/32-1);

singley = singley./(960/32-1);
doubley = doubley./(960/32-1);
tripley = tripley./(960/32-1);

singlez = singlez./(960/32-1);
doublez = doublez./(960/32-1);
triplez = triplez./(960/32-1);
%%

figure('Renderer', 'painters', 'Position', [50 50 900 600]);
% x
subplot(311);
plot(joints, singlex');
hold on;
plot(joints, doublex');
plot(joints, triplex');
legend('1 device','2 devices','3 devices','Location','Best');
title('x');
xlim([1 32]);
xticks(linspace(1,32,32));
xlabel('joint label (1-32)');
ylabel('position (mm)');
grid on;

% y
subplot(312);
plot(joints, singley');
hold on;
plot(joints, doubley');
plot(joints, tripley');
legend('1 device','2 devices','3 devices','Location','Best');
title('y');
xlim([1 32]);
xticks(linspace(1,32,32));
xlabel('joint label (1-32)');
ylabel('position (mm)');
grid on;


% z
subplot(313);
plot(joints, singlez');
hold on;
plot(joints, doublez');
plot(joints, triplez');
legend('1 device','2 devices','3 devices','Location','Best');
title('z');
xlim([1 32]);
xticks(linspace(1,32,32));
xlabel('joint label (1-32)');
ylabel('position (mm)');
grid on;

saveas(gcf,'x,y,z_positions.png');
