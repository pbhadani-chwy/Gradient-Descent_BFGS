% reading and preprocessing the input and the output
inp  = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sgx_train.jpg'));
inp = inp/255;
imshow(inp)
out = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sg_train.jpg'));
out_update = out(2:199,2:199);
%imshow(out_update);

% initializing the filter vector
f = rand(9,1);
out_update = double(reshape(out_update,[39204,1]));
out_update = double(out_update)./255.0;
%imshow(out_update);

% gradient decent formula
x = zeros(9,39204);
l = zeros;
count = 1;
for j=1:198
    for i = 1:198
        m = reshape(inp(i:i+2,j:j+2),[9,1]);
        x( :, count) = m;
        count = count +1;
    end
end
for i = 1: 2000
    
    y = f' * x;
    g = 1.0./(1.0 + exp(-y));
    sub = out_update' - g;
    g_diff = g.*(1-g);
    %g_diff = exp(-y)/(1 + 2.*exp(-y) + exp(-2.*y));
    %g_diff = exp(-y)./(1+exp(-y)) .^ 2);
    sub_res = sub.*g_diff;
    grad_res = -2/39204 * x * sub_res';
    
    L = 1/39204 * (sub * sub');
    l(i) = L;
    f = f - 0.1 .* grad_res;
end

test_img = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sgx_test.jpg'));
test_img = test_img./255.0;
%test_img = sigmoi

x1 = zeros(9,39204);
count = 1
for i=1:198
    for j = 1:198
        m1 = reshape(test_img(i:i+2,j:j+2),[9,1]);
        x1( :, count) = m1;
        count = count +1;
    end
end

fin_out = f' * x1;
fin_out = reshape(fin_out,[198,198]);

% testing in the train set
out = double(imread('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\sgx_train.jpg'));
out = out./255.0;
x2 = zeros(9,39204);
count = 1
for i=1:198
    for j = 1:198
        m2 = reshape(out(i:i+2,j:j+2),[9,1]);
        x2( :, count) = m2;
        count = count +1;   
    end
end
%fin_out = f' * x2;
%fin_out = reshape(fin_out,[198,198]);
%imshow(mat2gray(fin_out'.* 255.0));
imshow(mat2gray(fin_out'.* 255.0));
%imshow(test_img.* 255.0);
%plot(l);


%y = f.' * inp;
%z = out_update.' - 
