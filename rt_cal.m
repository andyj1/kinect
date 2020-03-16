% computes Rotation and Translation matrices from two pos+orientation streams
% transforms data stream 1 onto data stream 2 space
% pos1: secondary
% pos2 : main

clear all;clc;close all;
%% import data
n_joints = 32;
file1='joints_output.csv';
data1=xlsread(file1);

file2='joints_output2.csv';
data2=xlsread(file2);

%% extract positional and quarternion components
pos1 = data1(:,3:5);         pos2 = data2(:,3:5);
orientation1 = data1(:,6:9); orientation2 = data2(:,6:9);

%% 3dof data
figure('Position',[300 300 1600 600]);
% frame = 20; % indicate which frame to look at
for frame = 1:round(length(pos1)/32)
    startidx = frame*n_joints;
    [R_1,t_1] = arun(pos1(frame*n_joints+1:frame*n_joints+n_joints,:)',pos2(frame*n_joints+1:frame*n_joints+n_joints,:)');

    pos1_all = pos1(startidx+1:startidx+n_joints, 1:3);
    pos1_tran = R_1*pos1_all'+t_1;
    pos1_tran = pos1_tran';

    
    figure(1);
    subplot(121);
    scatter3(pos1(startidx+1:startidx+n_joints,1),pos1(startidx+1:startidx+n_joints,2),pos1(startidx+1:startidx+n_joints,3),'b'); hold on;
    scatter3(pos2(startidx+1:startidx+n_joints,1),pos2(startidx+1:startidx+n_joints,2),pos2(startidx+1:startidx+n_joints,3),'g'); hold off;
    title('original');
    legend('pos1','pos2');

    subplot(122);
    scatter3(pos1_tran(:,1),pos1_tran(:,2),pos1_tran(:,3),'m'); hold on;
    scatter3(pos2(startidx+1:startidx+n_joints,1),pos2(startidx+1:startidx+n_joints,2),pos2(startidx+1:startidx+n_joints,3),'g'); hold off;
    legend('pos1 tf','pos2');
    title('data 1 - transformed');
end

%% 6dof data
A=zeros(n_joints*4,4);
B=zeros(n_joints*4,4);
for i=1:n_joints
    Q1=quater2rot(orientation1(i,:));
    [U1,S1,V1] = svd(Q1);
    R1=U1*V1';
    t1=pos1(i,:)';
    A(4*i-3:4*i,:)=[R1 t1;
                    zeros(1,3) 1];
    Q2=quater2rot(orientation2(i,:));
    [U2,S2,V2] = svd(Q2);
    R2=U2*V2';
    t2=pos2(i,:)';
    B(4*i-3:4*i,:)=[R2 t2;
                    zeros(1,3) 1];
%     R21=R2'*R1;
%     t21(:,i)=-R2'*t1+R2'*t2;
    R21=R1*R2';
    t21(:,i)=-R1*R2'*t2+t1;

end

t_2=t21'
t_2_ave=mean(t_2)
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

    [d, n]=size(A)

    %Mean Center Data
    Ac = mean(A,2); % mean x,y,z for data stream 1
    Bc = mean(B,2); % mean x,y,z for data stream 2
    
    % find deviation from mean position (x,y,z)
    % ** A and B dimensions: 3x32 (= (x,y,z) by number of joints)
    A = A-repmat(Ac,1,n);
    B = B-repmat(Bc,1,n);
    
    % compute optimal rotation via SVD
    [u,s,v] = svd(A*B');
    size(A*B') % 3x3
    R = v*u';
    size(R) % 3x3
    if det(R)<0, disp('Warning: R is a reflection'); end

    %Calculate Optimal Translation
    t = Bc - R*Ac;
end
