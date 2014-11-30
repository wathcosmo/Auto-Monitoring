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
    
    frm1 = read(video_obj, 1);
    imshow(get_brightness_img(frm1));
    img = get_body(frm1);
    figure;
    imshow(img);
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
function img_out = get_body(frm)
    Fimg = get_foreground(frm);
    figure;
    imshow(Fimg);
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
    threshold = max(threshold, model.Miu(1) - 3 * model.Sigma(1));
    
    img_out = zeros(m, n);
    for x = 1 : m;
        for y = 1 : n;
            if Fimg(x,y) > threshold
                img_out(x,y) = 1;
            end
        end
    end
        
end

function fore_img = get_foreground(frm)
    global miu;
    global sigma;
    frm = get_brightness_img(frm);
    fore_img = 1 - frm ./ (miu + 3 * sigma);
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

