function detection
    fname = ['D:/study/Research/Image Processing/drosophila behavior/'...
        'videos/two-fly fighting/fight.avi_3.avi'];
    global video_obj;
    global nfrm;
    %global miu;
    video_obj = VideoReader(fname);
    frm = read(video_obj, inf);
    [m, n, ~] = size(frm);
    nfrm = video_obj.NumberOfFrames;
    if exist('bg_params.mat', 'file')
        load('bg_params.mat');
    else
        get_bg_param(m, n);  % calculate the mean(miu) and standard
                             % deviation(sigma) of background
    end
    %imshow(miu);
    for k = 785 : 786
        frm1 = read(video_obj, k);
        bimg = get_brightness_img(frm1);
%         figure;
%         imshow(bimg);
        Fimg = get_foreground(bimg);
%         figure;
%         imshow(Fimg);

        body = get_body(Fimg);
%         figure;
%         imshow(body);

        [num, body_states] = get_ellipse(body, Fimg);
        if num == 1
            fly_ctrs = double(body);
            for i = -1 : 1
                for j = -1 : 1
                    fly_ctrs(body_states(1).r + i, body_states(1).c + j) = 0.5;
                    fly_ctrs(body_states(2).r + i, body_states(2).c + j) = 0.5;
                end
            end
            figure;
            imshow(fly_ctrs);
        end
        wings = get_wings(Fimg);
        figure;
        imshow(wings);
    end
end

%%
function get_bg_param(m, n)
    global video_obj;
    global nfrm;
    global miu;
    global sigma;
    rand_arr = randperm(nfrm, 4000);
    
    miu = zeros(m, n);
    sigma = zeros(m, n);
    
    for k = 1 : 4000
        frm = read(video_obj, rand_arr(k));
        frm = get_brightness_img(frm);
        miu = miu + frm;
        sigma = sigma + frm .* frm; 
    end
    miu = miu / 4000;
    sigma = sqrt(sigma / 4000 - miu .* miu);
end

function img_out = get_brightness_img(frm)
    frm = im2double(frm);
    img_out = 0.4 * frm(:, :, 1) + 0.6 * frm(:, :, 2);
end

%%
function img_out = get_body(Fimg)
    [m, n] = size(Fimg);
    X = Fimg(:);
    [~, C] = kmeans(X, 3);
    [~, model] = gmm(X, C);
    
    for i = 1 : 2
        for j = i+1 : 3
            if model.Miu(i) < model.Miu(j)
                temp = model.Miu(i);
                model.Miu(i) = model.Miu(j);
                model.Miu(j) = temp;
                temp = model.Sigma(i);
                model.Sigma(i) = model.Sigma(j);
                model.Sigma(j) = temp;
                temp = model.Pi(i);
                model.Pi(i) = model.Pi(j);
                model.Pi(j) = temp;
            end
        end
    end
    
    funcb = [model.Miu(1), model.Sigma(1), model.Pi(1)];
    funca = [model.Miu(2), model.Sigma(2), model.Pi(2)];
    threshold = find_intersection(funca, funcb);
%     threshold = max(threshold, model.Miu(1) - 3 * model.Sigma(1));
    
    img_out = zeros(m, n);
    img_out(Fimg > threshold) = 1;
    img_out = bwareaopen(img_out, 256, 4);
end

function fore_img = get_foreground(bimg)
    global miu;
    global sigma;
    fore_img = 1 - bimg ./ (miu + 3 * sigma);
end

function sln = find_intersection(pa, pb)
    l = pa(1);
    r = pb(1);
    
    m = pa(1);
    s = pa(2);
    c = pa(3);
    fa = @(x) c / (sqrt(2*pi) * s) * exp(-(x-m)^2 / (2*s^2));
    
    m = pb(1);
    s = pb(2);
    c = pb(3);
    fb = @(x) c / (sqrt(2*pi) * s) * exp(-(x-m)^2 / (2*s^2));
    
    while (r - l) > 1e-3
        mid = (r + l) / 2;
        if fa(mid) > fb(mid)
            l = mid;
        else
            r = mid;
        end
    end
    sln = mid;
end

%%
function [lnum, body_sts] = get_ellipse(body, Fimg)
    [img_label, lnum] = bwlabel(body);
    
    if lnum == 1
        Fimg(body == 0) = 0;
        [row, col, val] = find(Fimg);
%         max_r = max(row);
%         min_r = min(row);
%         max_c = max(col);
%         min_c = min(col);
%         row = mapminmax(row');
%         col = mapminmax(col');
%         val = mapminmax(val');
%         X = [row', col', val'];
        X = [row, col, val];
        
        [~, C] = kmeans(X, 2);
        [~, model] = gmm(X, C);
        body_sts(1).r = round(model.Miu(1,1));
        body_sts(1).c = round(model.Miu(1,2));
        body_sts(2).r = round(model.Miu(2,1));
        body_sts(2).c = round(model.Miu(2,2));
        
%         body_sts(1).r = round(get_back(model.Miu(1,1), min_r, max_r));
%         body_sts(1).c = round(get_back(model.Miu(1,2), min_c, max_c));
%         body_sts(2).r = round(get_back(model.Miu(2,1), min_r, max_r));
%         body_sts(2).c = round(get_back(model.Miu(2,2), min_c, max_c));
%         body_sts(1).r = round(get_back(C(1,1), min_r, max_r));
%         body_sts(1).c = round(get_back(C(1,2), min_c, max_c));
%         body_sts(2).r = round(get_back(C(2,1), min_r, max_r));
%         body_sts(2).c = round(get_back(C(2,2), min_c, max_c));
    else
        body_sts = regionprops(img_label, 'Centroid', 'Orientation',...
            'MajorAxisLength', 'MinorAxisLength');
    end
    
end

function y = get_back(x, min, max)
    y = (x + 1) * (max - min) / 2 + min;
end

%%
function wings = get_wings(Fimg)
    threshs = multithresh(Fimg, 2);
    imth = imquantize(Fimg, threshs);
    wings = Fimg;
    wings(imth == 1) = 0;
    wings(imth == 2) = 0.5;
    wings(imth == 3) = 1;
end


