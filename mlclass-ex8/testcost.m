X=[0.1 0.4 0.7 0.8 0.5;0.2 0.5 0.8 0.7 0.4;0.3 0.6 0.9 0.6 0.3]';
Theta=[0.2 0.3 0.6;0.1 0.4 0.7;0.2 0.5 0.8]';
Y=[5 1 3;2 2 3;0 0 1;0 0 0;3 4 0];
R=[1 1 1;1 1 1;0 0 1;0 0 0;1 1 0];
num_user=3;
num_movie=5;
num_feature=3;
lambda=10;
param=[X(:);Theta(:)];
[J, grad]=cofiCostFunc(param,Y,R,num_user,num_movie,num_feature,lambda);
options = optimset('GradObj', 'on', 'MaxIter', 100);
ta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_user, num_movie, ...
                                num_feature, lambda)), ...
                param, options);

