% computes Rotation and Translation matrices from two pos+orientation streams
% transforms data stream 1 (sub) onto data stream 2 (main)
% SUBORDINATE --> MAIN data space

clear all;clc;close all;
%% import data
% extract positional and quarternion components

% data stream 1: SUBORDINATE 1
n_joints = 32;
% WINDOWS
% file1='joints_sync_2.csv';
% data1=xlsread(file1);
% mainpos1 = data1(:,3:5);         
% mainorientation1 = data1(:,6:9); 

% LINUX
maindata1 = readtable('joints_sync_1.csv','Format','auto','ReadRowNames',false);
mainpos1 = table2array(maindata1(:,3:5));
mainorientation1 = table2array(maindata1(:,6:9));

% data stream 2: MAIN
% WINDOWS
% file2='joints_sync_2.csv';
% subdata2=xlsread(file2);
% subpos2 = data2(:,3:5);
% suborientation2 = data2(:,6:9);

% LINUX
subdata2 = readtable('joints_sync_2.csv','Format','auto','ReadRowNames',false);
subpos2 = table2array(subdata2(:,3:5));
suborientation2 = table2array(subdata2(:,6:9));

%% if transformed joint points are available, visualize
% data stream 3: SUBORDINATE TRANSFORMED
subdata2_tf = readtable('joints_sync_tf.csv','Format','auto','ReadRowNames',false);
subpos2_tf = table2array(subdata2_tf(:,3:5));
suborientation2_tf = table2array(subdata2_tf(:,6:9));

f1 = figure('Position',[300 300 2100 600]);
for frame = 0:round(size(subpos2_tf,1)/32)-1
    startidx = frame*n_joints;
    
    % perform transformation here
    [R_1,t_1] = arun(subpos2(frame*n_joints+1:frame*n_joints+n_joints,:)',mainpos1(frame*n_joints+1:frame*n_joints+n_joints,:)');
    R_1
    t_1'
    subpos2_all = subpos2(startidx+1:startidx+n_joints, 1:3);
    subpos2_tran = R_1*subpos2_all'+t_1;
    subpos2_tran = subpos2_tran';

    figure(f1);
    subplot(131);
    scatter3(mainpos1(startidx+1:startidx+n_joints,1),mainpos1(startidx+1:startidx+n_joints,2),mainpos1(startidx+1:startidx+n_joints,3),'g'); hold on;
    scatter3(subpos2(startidx+1:startidx+n_joints,1),subpos2(startidx+1:startidx+n_joints,2),subpos2(startidx+1:startidx+n_joints,3),'b'); hold off;
%     legend('mainpos1 orig','subpos2 orig');
     title('original');

    subplot(132);
    scatter3(mainpos1(startidx+1:startidx+n_joints,1),mainpos1(startidx+1:startidx+n_joints,2),mainpos1(startidx+1:startidx+n_joints,3),'g'); hold on;
    scatter3(subpos2_tf(startidx+1:startidx+n_joints,1),subpos2_tf(startidx+1:startidx+n_joints,2),subpos2_tf(startidx+1:startidx+n_joints,3),'b'); hold off;
%     legend('mainpos1 orig','subpos2 tf');
    title('transformed');
    
    subplot(133);
    scatter3(mainpos1(startidx+1:startidx+n_joints,1),mainpos1(startidx+1:startidx+n_joints,2),mainpos1(startidx+1:startidx+n_joints,3),'g'); hold on;
    scatter3(subpos2_tran(:,1),subpos2_tran(:,2),subpos2_tran(:,3),'b'); hold off;
%     legend('mainpos1 orig','subpos2 transformed');
    title('supposed to be transformed like this...');
    
    sgtitle(sprintf("frame: %i",frame));
    
    pause(1);
    
end

%% 3dof data
figure('Position',[300 300 1600 600]);
% frame = 20; % indicate which frame to look at
for frame = 0:round(size(subpos2,1)/32)-1
    startidx = frame*n_joints;
    [R_1,t_1] = arun(subpos2(frame*n_joints+1:frame*n_joints+n_joints,:)',mainpos1(frame*n_joints+1:frame*n_joints+n_joints,:)');
    R_1
    t_1'
    subpos2_all = subpos2(startidx+1:startidx+n_joints, 1:3);
    subpos2_tran = R_1*subpos2_all'+t_1;
    subpos2_tran = subpos2_tran';

    
    figure(1);
    subplot(121);
    scatter3(mainpos1(startidx+1:startidx+n_joints,1),mainpos1(startidx+1:startidx+n_joints,2),mainpos1(startidx+1:startidx+n_joints,3),'g'); hold on;
    scatter3(subpos2(startidx+1:startidx+n_joints,1),subpos2(startidx+1:startidx+n_joints,2),subpos2(startidx+1:startidx+n_joints,3),'b'); hold off;
    legend('mainpos1 orig','subpos2 orig');
    title('original');

    subplot(122);
    scatter3(mainpos1(startidx+1:startidx+n_joints,1),mainpos1(startidx+1:startidx+n_joints,2),mainpos1(startidx+1:startidx+n_joints,3),'g'); hold on;
    scatter3(subpos2_tran(:,1),subpos2_tran(:,2),subpos2_tran(:,3),'m'); hold off;
    legend('mainpos1 orig','subpos2 transformed');
    title('transformed');
end

%% 6dof data
% A=zeros(n_joints*4,4);
% B=zeros(n_joints*4,4);
% for i=1:n_joints
%     Q1=quater2rot(suborientation1(i,:));
%     [U1,S1,V1] = svd(Q1);
%     R1=U1*V1';
%     t1=subpos1(i,:)';
%     A(4*i-3:4*i,:)=[R1 t1;
%                     zeros(1,3) 1];
%     Q2=quater2rot(suborientation2(i,:));
%     [U2,S2,V2] = svd(Q2);
%     R2=U2*V2';
%     t2=subpos2(i,:)';
%     B(4*i-3:4*i,:)=[R2 t2;
%                     zeros(1,3) 1];
% %     R21=R2'*R1;
% %     t21(:,i)=-R2'*t1+R2'*t2;
%     R21=R1*R2';
%     t21(:,i)=-R1*R2'*t2+t1;
% 
% end
% 
% t_2=t21'
% t_2_ave=mean(t_2)

%% Utility Functions
function R=quater2rot(Rq)

    i=Rq(1);
    j=Rq(2);
    k=Rq(3);
    r=Rq(4);
    R=[1-2*(j^2+k^2) 2*(i*j-k*r) 2*(i*k+j*r)
        2*(i*j+k*r) 1-2*(i^2+k^2) 2*(j*k-i*r)
        2*(i*k-j*k) 2*(j*k+i*r) 1-2*(i^2+j^2)
        ];
end

function [R,t] = arun(A,B)
    % Usage:: transforms A space onto B space
    %
    % Registers two sets of 3DoF data
    % Assumes A and B are d,n sets of data
    % where d is the dimension of the system 
    % typically d = 2,3
    % and n is the number of points
    % typically n>3
    %
    % Mili Shah
    % July 2014

    [d, n]=size(A);

    %Mean Center Data
    Ac = mean(A,2); % mean x,y,z for data stream 1
    Bc = mean(B,2); % mean x,y,z for data stream 2
    
    % find deviation from mean position (x,y,z)
    % ** A and B dimensions: 3x32 (= (x,y,z) by number of joints)
    A = A-repmat(Ac,1,n);
    B = B-repmat(Bc,1,n);
    
    % compute optimal rotation via SVD
    [u,s,v] = svd(A*B');
    u
    v
%     size(A*B') % 3x3
    R = v*u';
%     size(R) % 3x3
    if det(R)<0, disp('Warning: R is a reflection'); end

    %Calculate Optimal Translation
    t = Bc - R*Ac;
end
