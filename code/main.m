%% Initialize

clc; clear all;

run('vlfeat-0.9.20-bin/vlfeat-0.9.20/toolbox/vl_setup')

%% Taking parnoramic pictures
image1 = imresize(imread('../images/picture1_1.jpg'),[256,256]);
image2 = imresize(imread('../images/picture1_2.jpg'),[256,256]);
image3 = imresize(imread('../images/picture1_3.jpg'),[256,256]);
image4 = imresize(imread('../images/picture1_4.jpg'),[256,256]);
image5 = imresize(imread('../images/picture1_5.jpg'),[256,256]);

im1 = single(rgb2gray(image1));
im2 = single(rgb2gray(image2));
im3 = single(rgb2gray(image3));
im4 = single(rgb2gray(image4));
im5 = single(rgb2gray(image5));
image1 = single(image1);
image2 = single(image2);
image3 = single(image3);
image4 = single(image4);
image5 = single(image5);

%% Feature Extraction
[f1,d1] = vl_sift(im1);
[f2,d2] = vl_sift(im2);
[f3,d3] = vl_sift(im3);
[f4,d4] = vl_sift(im4); 
[f5,d5] = vl_sift(im5);

%% Getting homography estimation function using RANSAC
H12 = HbyRANSAC(im2,im1);
H23 = HbyRANSAC(im3,im2);
H43 = HbyRANSAC(im3,im4);
H54 = HbyRANSAC(im4,im5);

H = zeros(3,3,4);
H(:,:,1) = H23*H12;
H(:,:,2) = H23;
H(:,:,3) = H43;
H(:,:,4) = H43*H54;

%% Warping images

x1=[1;1;1];
x2=[1;255;1];
x3=[255;1;1];
x4=[255;255;1];
x_warp = zeros(3,4,4);
for i=1:4
    x1_warp = H(:,:,i)*x1;
    x1_warp = x1_warp/x1_warp(3);
    x2_warp = H(:,:,i)*x2;
    x2_warp = x2_warp/x2_warp(3);
    x3_warp = H(:,:,i)*x3;
    x3_warp = x3_warp/x3_warp(3);
    x4_warp = H(:,:,i)*x4;
    x4_warp = x4_warp/x4_warp(3);
    x_warp(:,1,i)=x1_warp;
    x_warp(:,2,i)=x2_warp;
    x_warp(:,3,i)=x3_warp;
    x_warp(:,4,i)=x4_warp;
end
minmax_window = minmax(horzcat([1;1;1],[255;255;1],x_warp(:,:,1),x_warp(:,:,2),x_warp(:,:,3),x_warp(:,:,4)));
window_x = round(minmax_window(1,1));
window_y = round(minmax_window(2,1));

x_range = [minmax_window(1,1) , minmax_window(1,2)]
y_range = [minmax_window(2,1) , minmax_window(2,2)]


T1 = maketform('projective',H(:,:,1)');
I1_warp = imtransform(image5,T1,'XData', x_range ,'YData', y_range);
T2 = maketform('projective',H(:,:,2)');
I2_warp = imtransform(image4,T2,'XData', x_range ,'YData', y_range);
T3 = maketform('projective',eye(3));
I3_warp = imtransform(image3,T3,'XData', x_range ,'YData', y_range);
T4 = maketform('projective',H(:,:,3)');
I4_warp = imtransform(image2,T4,'XData', x_range ,'YData', y_range);
T5 = maketform('projective',H(:,:,4)');
I5_warp = imtransform(image1,T5,'XData', x_range ,'YData', y_range);

I_panorama= max(I1_warp,max(I2_warp,max(I3_warp,max(I5_warp,I4_warp))));

%% Display results

% show extracted features 
hFig1 = figure(5);
set(hFig1, 'Position', [160 140 1600 800])

subplot(1,5,1);
imshow(im1/255);
perm = randperm(size(f1,2));
sel = perm(1:min(200,size(f1,2)));
h1 = vl_plotframe(f1(:,sel));
h2 = vl_plotframe(f1(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);

subplot(1,5,2);
imshow(im2/255);
perm = randperm(size(f2,2));
sel = perm(1:min(200,size(f2,2)));
h1 = vl_plotframe(f2(:,sel));
h2 = vl_plotframe(f2(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);

subplot(1,5,3);
imshow(im3/255);
perm = randperm(size(f3,2));
sel = perm(1:min(200,size(f3,2)));
h1 = vl_plotframe(f3(:,sel));
h2 = vl_plotframe(f3(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);

subplot(1,5,4);
imshow(im4/255);
perm = randperm(size(f4,2));
sel = perm(1:min(200,size(f4,2)));
h1 = vl_plotframe(f4(:,sel));
h2 = vl_plotframe(f4(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);

subplot(1,5,5);
imshow(im5/255);
perm = randperm(size(f5,2));
sel = perm(1:min(200,size(f5,2)));
h1 = vl_plotframe(f5(:,sel));
h2 = vl_plotframe(f5(:,sel));
set(h1,'color','k','linewidth',3);
set(h2,'color','y','linewidth',2);

figure(6)
imshow(I_panorama/255);


