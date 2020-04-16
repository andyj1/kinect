
clear all;clc;close all;
%% import data
n_joints = 32;

avgdata = readtable('./saved_data/joints_gen_avg.csv','Format','auto','ReadRowNames',false);
avgpos = table2array(avgdata(:,3:5));
avgorientation = table2array(avgdata(:,6:9));

sub1data = readtable('./saved_data/joints_gen_sub1.csv','Format','auto','ReadRowNames',false);
sub1pos = table2array(sub1data(:,3:5));
sub1orientation = table2array(sub1data(:,6:9));

sub2data = readtable('./saved_data/joints_gen_sub2.csv','Format','auto','ReadRowNames',false);
sub2pos = table2array(sub2data(:,3:5));
sub2orientation = table2array(sub2data(:,6:9));

masterdata = readtable('./saved_data/joints_gen_master_orig.csv','Format','auto','ReadRowNames',false);
masterpos = table2array(masterdata(:,3:5));
masterorientation = table2array(masterdata(:,6:9));

%%
vidfile1 = VideoWriter('scatters.avi');
open(vidfile1);

f1 = figure('Position',[300 300 900 600]);
for frame = 1:min(size(sub1pos,1),size(sub2pos,1))/n_joints-1
    startidx = frame*n_joints;
    scatter3(sub1pos(startidx+1:startidx+n_joints,1),sub1pos(startidx+1:startidx+n_joints,2),sub1pos(startidx+1:startidx+n_joints,3),'g*');
    hold on;
    scatter3(sub2pos(startidx+1:startidx+n_joints,1),sub2pos(startidx+1:startidx+n_joints,2),sub2pos(startidx+1:startidx+n_joints,3),'b*');
    scatter3(avgpos(startidx+1:startidx+n_joints,1),avgpos(startidx+1:startidx+n_joints,2),avgpos(startidx+1:startidx+n_joints,3),'ro'); 
    scatter3(masterpos(startidx+1:startidx+n_joints,1),masterpos(startidx+1:startidx+n_joints,2),masterpos(startidx+1:startidx+n_joints,3),'mo');
    legend('sub1','sub2','avg','master');
    title(sprintf("Frame: %i",frame));
    xlabel('x');ylabel('y');zlabel('z');
    axis([-200 400 -400 1000 100 800]);
    hold off;
    
    writeVideo(vidfile1,getframe(gcf));
    
    pause(1);
end

close(vidfile1);
