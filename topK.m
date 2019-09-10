function ratio = topK(Y, pred)
% Y: groud-truth label matrix
% pred: the prediced label matrix
% This function returns P@k with k specified as 1, 3, and 5.

topK = [1, 3, 5];

ratio = zeros(length(topK), 1);

for i = 1: length(topK)
    
    count = 0;
    for j = 1: topK(i)
        for k = 1: size(pred, 1)
            if pred(k, j) > 0 && Y(k, pred(k, j)) == 1
                count = count + 1;
            end
        end
    end
    
    ratio(i) = count / ( topK(i) * size(Y, 1) );
    
end