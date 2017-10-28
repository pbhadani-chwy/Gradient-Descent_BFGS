% reading and preprocessing the input and the output
inp  = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sgx_train.jpg'));
inp = inp./255.0;
out = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sg_train.jpg'));
out_update = out(2:199,2:199);
out_update = double(reshape(out_update,[39204,1]));
out_update = double(out_update)./255.0;

% initializing the filter vector
f_update = ones(9,1);

% gradient decent formula
x = zeros(9,39204);
grad_update = zeros(9,1);
count = 1;
for j=1:198
    for i = 1:198
        m = reshape(inp(i:i+2,j:j+2),[9,1]);
        x( :, count) = m;
        count = count +1;
    end
end
l = zeros;
g =  eye(3 * 3);
f_entry = f_update;
grad_entry = grad_update;
y = f_entry' * x;
g_1 = 1.0./(1.0 + exp(-y));
sub = out_update' - g_1;
g_diff = g_1.*(1-g_1);
%g_diff = exp(-y)/power(1+exp(-y),2);
sub_res = sub.*g_diff;
grad_update = -2/39204* x * sub_res';
grad_update = -grad_update;
f_update = f_update - 0.00001 * grad_update;

for i = 1:1000
    %disp(i);
    f_entry = f_update;
    grad_entry = grad_update;
    y = f_entry' * x;
    g_1 = 1.0./(1.0 + exp(-y));
    sub = out_update' - g_1;
    g_diff = g_1 .*(1-g_1);
    %g_diff = exp(-y)/power(1+exp(-y),2);
    sub_res = sub.*g_diff;
    grad_update = -2/39204* x * sub_res';
    grad_update = -grad_update;
    % calculation of hessian matrix
    
    f_update = f_entry - 0.00001 * g * grad_update;
    %disp(f_update);
    p = f_update - f_entry;
    %p = (1 / (1 + exp(-p)))';
    v = grad_update - grad_entry;
    %v = (1 / (1 + exp(-v)))';
    u = p/(p'*v) - ((g*v)/(v'*g*v));
    
    g = g + ((p*p')/(p.'*v)) - ((g*v)*v'*g)/(v'*g*v) + (v'*g*v)*(u*u');

    %u = (1 / (1 + exp(-u)))';
    L = (sub * sub')/39204.0;
    l(i) = L;
end



test_img = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sgx_test.jpg'));
test_img = test_img/225.0;

x1 = zeros(9,39204);
count = 1
for i=1:198
    for j = 1:198
        m1 = reshape(test_img(i:i+2,j:j+2),[9,1]);
        x1( :, count) = m1;
        count = count +1;
    end
end

%fin_out = f_update.' * x1;
%fin_out = reshape(fin_out,[198,198])'.* 255.0;

% testing on the training data

fin_out = f_update' * x;
fin_out = reshape(fin_out,[198,198])'.* 255.0;



imshow(mat2gray(fin_out'));
%plot(l);
%imshow(test_img.* 255.0);



