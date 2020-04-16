% Author: Andy Jeong
% Last Updated: November 27, 2019

clear all; close all; clc;
%% Jitter estimate
% compute average difference in joint position estimates
% for a stationary object (e.g. mannequin)

data_orig = readtable('../1-joints_100frames_mannequin.csv');
data = table2array((data_orig(:,1:3)));

average = mean(data,1);
avg_diff = mean(data - repmat(average, [length(data), 1]), 1);
