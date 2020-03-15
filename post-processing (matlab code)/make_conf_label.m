function [confidence] = make_conf_label(data)
    confidence = zeros([height(data), 1]);
    for i = 1:height(data)
       entry = string(data(i,4).Var4);
       if strcmp(entry, "Medium confidence in joint pose")
           confidence(i) = 3;
       elseif strcmp(entry, "The joint is not observed(likely due to occlusion) - predicted joint pose")
           confidence(i) = 2;
       elseif strcmp(entry, "The joint is out of range(too far from depth camera)")
           confidence(i) = 1;
       end
    end
end