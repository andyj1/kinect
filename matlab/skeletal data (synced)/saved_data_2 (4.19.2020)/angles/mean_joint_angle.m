clc;clear all;close all;

%% read in the data
onedev = readtable('./joints_single_angles.csv'); onedev = onedev(:,2:13);
twodev = readtable('./joints_sync_angles.csv'); twodev = twodev(:,2:13);
threedev = readtable('./joints_gen_angles.csv'); threedev = threedev(:,2:13);

onedev = table2array(onedev);
twodev = table2array(twodev);
threedev = table2array(threedev);
%% take average
onemean = mean(onedev);
twomean = mean(twodev);
threemean = mean(threedev);
%% plot joint angles from 1,2,3 device settings
angles = {'A','B','C','D','E','F','G','H','I','J','K','L'};
f = figure('Renderer', 'painters', 'Position', [50 50 900 600]);
for i=1:length(angles)
    x = 1:length(angles);
    plot(x, onemean, x, twomean, x, threemean);
    xlim([1, 12]);
    xticklabels(angles);
    ylabel('Degrees');
    xlabel('Joint Angle');
    legend('1 device','2 devices','3 devices');
end
sgtitle('average joint angles with 1,2,3 devices for 12 joint angles');
saveas(gcf,'./mean joint angles.png');