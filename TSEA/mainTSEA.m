clear;clc;
% A two-stage evolutionary algorithm based on image clustering for hyperspectral endmember extraction
% Input:
%   image_3d:      The 3-D matrix of the image data (reflectance)
%   X:             The 2-D matrix of the image data
%   L:             Number of bands
%   N:             Number of pixels
%   row:           Number of rows for the 3-D image matrix
%   col:           Number of columns for the 3-D image matrix
%   P:             Number of endmembers
%   pop_num:       Size of population
%   maxevaluation: Maximum evaluation times of objective function
%   e:             Scaling factor in the calculation of Div
%   proC:          Crossover rate of SBX
%   disC:          Distribution index of SBX
%   proM:          Mutation rate of the polynomial mutation
%   disM:          Distribution index of the polynomial mutation
% Output:
%   x:             The final population
%   f_x:           The value of objective functions

%%******************************* Read data and set parameters *******************************%%
% Input the Urban image
load("F:\Urban\Urban_162.mat")
image_2d = Y./1000;
X = image_2d;
[L,N] = size(image_2d);
row = 307;
col = 307;
image_3d = zeros(row,col,L);
for i = 1:L
    image_3d(:,:,i) = reshape(image_2d(i,:),row,col);
end
P = 6;   % Number of endmembers

% %Input the Berlinsub1 image
% [img2d,img3d] = freadenvi('D:\BerlinUrban2009\BerlinSub1');
% X = img2d';
% [L,N] = size(X);
% row = size(img3d,1);
% col = size(img3d,2);
% for i = 1:L
%     image_3d(:,:,i) = reshape(X(i,:),row,col);
% end
% P = 6;


% Set parameters
pop_num = 20;                         % Size of population
maxevaluation = 6000;                 % Maximum evaluation times of objective function
e = 1*10^-6;                          % Scaling factor in the calculation of Div
proC = 1;                             % Crossover rate of SBX
disC = 20;                            % Distribution index of SBX
proM = 1;                             % Mutation rate of the polynomial mutation
disM = 20;                            % Distribution index of the polynomial mutation
PS_record1 = zeros(maxevaluation,1);  % Minimum volume inverse for each iteration
PS_record2 = zeros(maxevaluation,1);  % Minimum RMSE for each iteration
stagelist = [];                       % Recording of the stage

% Reduce the dimension of the hyperspectral image
trans = dim_reduction(P,X,image_3d);
X_reduction = trans * X;

%%******************************* Main parts of TSEA *******************************%%
tic;
% Kmeans cluster
X = X';
class = kmeans(X,P);
classnum = zeros(1,P);
newX = [];
lower = zeros(1,P);
upper = zeros(1,P);
for i = 1:P
    classnum(1,i) = sum(class==i);
    newX = [newX;X(class==i,:)];
    lower(i) = sum(classnum(1,1:i-1))+1;
    upper(i) = lower(i)+classnum(1,i)-1;
end
X = newX';
X_reduction = trans * X;
% Initialization
x = zeros(pop_num,P);
for j = 1:pop_num
    x(j,:) = floor(rand(1,P).*classnum)+lower;
end
evaluation = 0;
for j=1:pop_num
    f_x(j,1) = 1/volume(X_reduction(:,x(j,:)));
    f_x(j,2) = unmixed(X,X(:,x(j,:)),1);
    evaluation = evaluation + 1;
end
% Calculate the non-dominated front number of each solution
[FrontNo,MaxFNo] = NDSort(f_x,inf);
% Normalization
f_x2 = f_x;
% Calculate the convergence degree of each solution in x
Con = sum(f_x2,2);
% Calculate the diversity degree of each solution in P
dist = zeros(pop_num,pop_num);
for i = 1:pop_num
    for j = 1:pop_num
        if i == j
            dist(i,j) = inf;
        else
            dist(i,j) = norm(f_x2(i,:)-f_x2(j,:));
        end
    end
end
dist = sort(dist,2);
Div = dist(:,1)+e*dist(:,2);

% Main Loop
iter = 0;
while evaluation < maxevaluation
    %% Determining stage
    if MaxFNo > 1
        stage = 1;
    else
        stage = 2;
    end
    stagelist = [stagelist stage];
    %% Mating selection
    candidate = randperm(pop_num,4);
    if stage == 1
        if FrontNo(candidate(1)) < FrontNo(candidate(2))
            parent1 = x(candidate(1),:);
        elseif FrontNo(candidate(1)) > FrontNo(candidate(2))
            parent1 = x(candidate(2),:);
        elseif Con(candidate(1)) < Con(candidate(2))
            parent1 = x(candidate(1),:);
        else
            parent1 = x(candidate(2),:);
        end
        if FrontNo(candidate(3)) < FrontNo(candidate(4))
            parent2 = x(candidate(3),:);
        elseif FrontNo(candidate(3)) > FrontNo(candidate(4))
            parent2 = x(candidate(4),:);
        elseif Con(candidate(3)) < Con(candidate(3))
            parent2 = x(candidate(3),:);
        else
            parent2 = x(candidate(4),:);
        end
    else
        if Con(candidate(1)) < Con(candidate(2))
            parent1 = x(candidate(1),:);
        else
            parent1 = x(candidate(2),:);
        end
        if Div(candidate(3)) > Div(candidate(4))
            parent2 = x(candidate(3),:);
        else
            parent2 = x(candidate(4),:);
        end
    end
    %% Produce offspring
    % Simulated binary crossover (SBX)
    off = x;
    idx = randperm(size(off,1),size(off,1));
    x1 = parent1;
    x2 = parent2;
    n = size(x1,1);
    beta = zeros(n,P);
    mu   = rand(n,P);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],n,P);
    beta(rand(n,P)<0.5) = 1;
    beta(repmat(rand(n,1)>proC,1,P)) = 1;
    off = round([(x1+x2)/2+beta.*(x1-x2)/2
        (x1+x2)/2-beta.*(x1-x2)/2]);
    % Polynomial mutation
    Lower = repmat(lower,2*n,1);
    Upper = repmat(upper,2*n,1);
    Site  = rand(2*n,P) < proM/P;
    mu    = rand(2*n,P);
    temp  = Site & mu<=0.5;
    off       = min(max(off,Lower),Upper);
    off(temp) = round(off(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(off(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1));
    temp = Site & mu>0.5;
    off(temp) = round(off(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-off(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))));
    off = off(randperm(2,1),:);
    % De-duplication
    temp = unique(off);
    if length(temp) < P
        pos = Xmin:Xmax;
        pos(temp) = [];
        num = randperm(length(pos),P-length(temp));
        temp = [temp pos(num)];
    end
    off = temp;
    f_off(1,1) = 1/volume(X_reduction(:,off));
    f_off(1,2) = unmixed(X,X(:,off),1);
    evaluation = evaluation + 1;
    % Normalization
    f_off2 = f_off;
    Con_off = sum(f_off2);
    dist_off = zeros(1,pop_num);
    for j = 1:pop_num
        dist_off(1,j) = norm(f_x2(j,:)-f_off2);
    end
    dist_off = sort(dist_off);
    Div_off = dist_off(1,1)+e*dist_off(1,2);
    %% EnvironmentalSelection
    if size(unique([x;off],'rows'),1) <= pop_num
        
    elseif stage == 1
        x = [x;off];
        f_x = [f_x;f_off];
        Con = [Con;Con_off];
        Div = [Div;Div_off];
        [FrontNo,MaxFNo] = NDSort(f_x,inf);
        XX = find(FrontNo==MaxFNo);
        temp = Con(XX,:);
        w = find(temp==max(temp));
        w = XX(w(1));
        x(w,:) = [];
        f_x(w,:) = [];
        Con(w,:) = [];
        Div(w,:) = [];
    else
        w = find(dist_off==min(dist_off));
        w = w(1);
        if Con_off <= Con(w) && Div_off >= Div(w)
            x = [x;off];
            f_x = [f_x;f_off];
            Con = [Con;Con_off];
            Div = [Div;Div_off];
            x(w,:) = [];
            f_x(w,:) = [];
            Con(w,:) = [];
            Div(w,:) = [];
        end
    end
    [FrontNo,MaxFNo] = NDSort(f_x,inf);
    iter = iter +1;
    PS_record1(iter) = min(f_x(:,1));
    PS_record2(iter) = min(f_x(:,2));
    f_x
    fprintf('The %dth iteration finished\n',iter);
end
t = toc;
time = t;