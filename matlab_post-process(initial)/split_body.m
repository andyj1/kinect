% Author: Andy Jeong
% Last Updated: November 26, 2019

%%
data = readtable('joints_output.csv', 'ReadVariableNames',false, 'Delimiter', ',');

% ---------------------------------
% set the number of bodies detected
num_bodies = 1;
% ---------------------------------

for i = 1:num_bodies
    bid = str2double(string(data.Var1));
    body_data = data(find(bid == i),:);
    body_data.Var3

    x = data.Var3;
    y = data.Var4;
    z = data.Var5;
    ci = data.Var10;

    table_data = table(x,y,z,ci);
    writetable(table_data, sprintf('body_%d.csv', i));
end
