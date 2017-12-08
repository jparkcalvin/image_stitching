function H = HbyRANSAC(im_dst, im_src)

t = 1.25;
p = 0.99;

%% Find matched points
[f1,d1] = vl_sift(im_dst) ;
[f2,d2] = vl_sift(im_src) ;

[matches, scores] = vl_ubcmatch(d1,d2) ;

n_Matches = size(matches,2) ;

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

%% RANSAC with the adaptive determination method for the # of samples N
clear H score ok ;
N = inf;
sampleCount = 0;
while N > sampleCount
    t = sampleCount + 1;
  % estimate homograpyh
  subset = vl_colsubset(1:n_Matches, 4) ;
  A = [] ;
  for i = subset
    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
  end
  [U,S,V] = svd(A) ;
  H{t} = reshape(V(:,9),3,3) ;

  % score homography
  X2_ = H{t} * X1 ;
  du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
  dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
  ok{t} = (du.*du + dv.*dv) < 6*6 ;
  score(t) = sum(ok{t}) ;
  
  epsilon = 1-(score(t)/n_Matches);
  N = log(1-p)/log(1-(1-epsilon)^4);
  sampleCount = sampleCount + 1;
    
end

[score, best] = max(score) ;
H = H{best} ;
ok = ok{best} ;

%% Display matched points
dh1 = max(size(im_src,1)-size(im_dst,1),0) ;
dh2 = max(size(im_dst,1)-size(im_src,1),0) ;

figure ; clf ;
subplot(2,1,1) ;
imagesc([padarray(im_dst,dh1,'post') padarray(im_src,dh2,'post')]) ;
o = size(im_dst,2) ;
line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
     [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
title(sprintf('%d putative matches', n_Matches)) ;
axis image off ;

subplot(2,1,2) ;
imagesc([padarray(im_dst,dh1,'post') padarray(im_src,dh2,'post')]) ;
o = size(im_dst,2) ;
line([f1(1,matches(1,ok));f2(1,matches(2,ok))+o], ...
     [f1(2,matches(1,ok));f2(2,matches(2,ok))]) ;
title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
              sum(ok), ...
              100*sum(ok)/n_Matches, ...
              n_Matches)) ;
axis image off ;

drawnow ;

end

