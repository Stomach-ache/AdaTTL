function [y, model_size, test_time] = sampling(data, curK, instance_cand, label_cand)

ni = length(instance_cand);
nl = length(label_cand);

y = zeros(3, ni * nl);
time_used = ones(1, ni * nl);
test_time = ones(1, ni * nl);
model_size = ones(1, ni * nl);

numLabels = size(data.Y, 2);
numInstance = size(data.Y, 1);

% sort labels from the least frequent to the most frequent
[~, positions] = sort(sum(data.Y, 1));


for ii = 1: ni
    for jj = 1: nl
        n = instance_cand(ii) * numInstance / 100;
        l = label_cand(jj);
        
        % random sampling instances
        perm = randperm(numInstance);
        perm = perm(1:n);
        
        i = floor(l * numLabels / 100);
        if (i == 0)
            i = 1;
        end
        ind = positions(i:numLabels);

        tmpY  = data.Y;
        tmpYt = data.Yt;
        tmpX  = data.X;
        
        data.Y  = data.Y(:, ind);
        data.Yt = data.Yt(:, ind);
        data.Y  = data.Y(perm, :);
        data.X = data.X(perm, :);

        
        % train model
        % LMLL could be replaced by another other models
        tic;
        [W, H, wall_time] = train_ml(data.Y, data.X, data.Yt, data.Xt, ['-k ', int2str(curK)]);
        time_used((ii - 1) * nl + jj) = toc;
        
        
    
        % calculate used memory space
        curModelSize = 0;
        tmp = whos('W');
        curModelSize = curModelSize + tmp.bytes;
        tmp = whos('H');
        curModelSize = curModelSize + tmp.bytes;
    
        model_size((ii - 1) * nl + jj) = curModelSize;

        tic
        for j = 1: size(data.Yt, 1)
            pred = H' * (W * data.Xt(j, :)');
            [~, pos] = sort(pred, 'descend');
            result.predictLabels(j, :) = pos(1:5);
        end
    
        test_time((ii - 1) * nl + jj) = toc;

        result.precision = topK(data.Yt, result.predictLabels);
        y(:, (ii - 1) * nl + jj) = result.precision;
   
        data.X = tmpX;
        data.Y = tmpY;
        data.Yt = tmpYt;

    end
end
