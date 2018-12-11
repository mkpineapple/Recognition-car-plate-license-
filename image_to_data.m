% @author: jiryi for final project
% @contributor: qiqi and Ke Ma
function image_to_data()
%% This file is to pre-processing the car images dataset
% this file will generate a numerical respresentation of the data set
% with the relation: feature data & label data
%
% usage: image_to_data()
% 
% for each image, we do the following
%     1. segement the letter or numbers from the plate, 
%        make sure that the letter images and number images have the same size, 
%        example: for a car plate with 7 characters, then we should have 7 images
%     2. turn the 7 small images, we convert them to be gray images
%        we won't work on RGB images
%     3. use one vector to represent each image,
%        example: for a small image of size (N_1,N_2), we use an vector of size N_1*N_2 to represent it
%     4. we give a label for each of the 7 small images, 
%        and represent it via a one-hot vector of size 36 (totally 26 letters, 10 numbers),
%        the first 10 entries correspond to number 0,1,...,10,
%        the rest 26 entries correspond to letter A,B,...,Z
%        example: if the image has the "0", 
%        then the vector of size 36 to represent it will have 1 in the first entry and zeros elsewhere
%     5. we will let the 7 vectors representing the 7 characters in one plate be together,
%        the order of the seven vectors will be the same as the order as they show in the plate
%     6. we will process at least 150 plate images (150*7 small images), 
%        80% of the 100 plate images (80) will be used to train the network
%        the rest 20% (20) will be used as testing data set
%     7. character and index key: character <-> index
%         0 <-> 1, 1 <-> 2, 2 <-> 3, 3 <-> 4
%         4 <-> 5, 5 <-> 6, 6 <-> 7, 7 <-> 8
%         8 <-> 9, 9 <-> 10, 
%         A <-> 11, B <-> 12, C <-> 13, D <-> 14
%         E <-> 15, F <-> 16, G <-> 17, H <-> 18
%         I <-> 19, J <-> 20, K <-> 21, L <-> 22
%         M <-> 23, N <-> 24, O <-> 25, P <-> 26
%         Q <-> 27, R <-> 28, S <-> 29, T <-> 30
%         U <-> 31, V <-> 32, W <-> 33, X <-> 34
%         Y <-> 35, Z <-> 36
%     
% attention
%     (1) I have used P6070001~P6070100, P9180001~P9180050
%     (2) Only consider the plate with 7 characters
% 
%
% The outputs of this file will be
%     P607_50_Plate_Character_Labels.mat: the label for each character in the plate
%                                         consecutive 7 charactars form a plate number sequence
%                                         i.e., label for A, label for B, label for C,...
% 
%     P607_50_Plate_Labels: the plate number sequence, 
%                           i.e., ABC1314
% 
%     P607_50_Plates_Character_Images: the image of each character showing in the plate
%                                      i.e., image of A, image of B, ...
% 
%     P607_50_Plates_Images: the image of the whole plate, i.e., image of ABC1314 
%
% attention:
% (1) image names should be continuous
% (2) N_plate should be equal to the number of cells in excel file

% by JYI on 11/05/2018

%% feature data construction (attention)
% every 7 consecutive samples are combined to be an plate number sequence
% 
% every plate consists of 7 character images
% 
% each character image has size 40 by 20

fprintf('Now construct feature data\n');
N_plate = 60; % need modifications according to number of images
N_class = 36; L_plate = 7;
L_high = 40; L_wide = 20; L_feat = L_high*L_wide;
I_high = 320; I_wide = 240; I_pix = I_high*I_wide;

%%
% data_feat = zeros(N_plate*L_plate, L_feat);
% data_img = zeros(N_plate, I_pix); 
data_feat = [];
data_img = [];
for j=1:N_plate
    
    if j < 10 % whole image construction
        imgfile = sprintf('P607000%d.jpg',j);
    else
        imgfile = sprintf('P60700%d.jpg',j);
    end
    cimg = imread(imgfile);
    cimg = rgb2gray(cimg);
    cimg = imresize(cimg,[I_high,I_wide]);
    cimg = reshape(cimg,1,I_pix);
    data_img = [data_img; cimg];
    
%     ck_cimg = reshape(data_img(j,:),I_high,I_wide);
%     imshow(ck_cimg);
%     imshow(cimg);
%     imshow(reshape(cimg,I_high,I_wide));
    
    
    for i=1:L_plate % character construction
        if j < 10
            ifile = sprintf('P607000%d-c%d.jpg',j,i);
        else
            ifile = sprintf('P60700%d-c%d.jpg',j,i);
        end
        
        img = imread(ifile);
        img = rgb2gray(img);
        img = imresize(img,[L_high,L_wide]);
        img = reshape(img,1,L_feat);
        data_feat = [data_feat;img];
    end

end

save('Plates_Images_5.mat','data_img');
save('Plates_Character_Images_5.mat','data_feat');

%% feature data check
fprintf('Now perform feature data checking\n');

ind_ck = randsample(N_plate,1);
ind_ch = (ind_ck-1)*L_plate;

% show plate character images
figure; 
for i=1:L_plate
    subplot(1,7,i);
    ind_ch = ind_ch + 1;
    imshow(reshape(data_feat(ind_ch,:),L_high,L_wide))
end

% show the whole plate
img = reshape(data_img(ind_ck,:),I_high,I_wide);
figure; imshow(img);

% answer = questdlg('Make sure you have checked the feature data instance. Proceed?',...
%                   'Feature data checking',...
%                   'Yes','No','Yes');
% switch answer
%     case 'Yes'
%         fprintf('Feature data checking done\n');
%     case 'No'
%         exit; 
% end

%% label data construction (attention)
fprintf('Now construct label data\n');
[~,data_cha] = xlsread('label.xlsx','B2:B61'); % need modifications according to the excel file label.xls
%%
% the N_plate should be equal to the number of excel cells
N_class = 36;
L_plate = 7;

data_lab = [];
for i=1:N_plate
    lab = data_cha(i); lab = lab{1};
    
    for j=1:L_plate
        cha = lab(j);
        cha_v = label_to_vector(cha,N_class);
        % cha_v = @label_to_vector;
        % cha_v = label_to_vector;
        data_lab = [data_lab; cha_v];
    end
    
end

save('Plate_Labels_5','data_cha');
save('Plate_Character_Labels_5.mat','data_lab');

%% label data check
fprintf('Now perform label data checking\n'); 
ind_ck = randsample(N_plate,1);
lab_ck = data_cha(ind_ck);
lab_ck = lab_ck{1}; 
fprintf('The plate reads as: %8s\n',lab_ck);

ind_ch = (ind_ck - 1)*L_plate;
lab_arr = [];
for i=1:L_plate
    ind_ch = ind_ch + 1;
    vec_lab = data_lab(ind_ch,:);
    lab = vector_to_label(vec_lab,N_class);
    lab_arr = [lab_arr, num2str(lab)];   
end

fprintf('The label sequence of plate is :\n'); lab_arr

% answer = questdlg('Make sure you have checked the label data instance. Proceed?',...
%                   'Label data checking',...
%                   'Yes','No');
% switch answer
%     case 'Yes'
%         fprintf('Label data checking done\n');
%     case 'No'
%         exit; 
% end

%% feature data and label data correspondence check
fprintf('Now performa feature and label correspondence check\n'); 

ind_ck = randsample(N_plate,1);
lab_ck = data_cha(ind_ck); lab_ck = lab_ck{1};
fprintf('The plate reads as: %s\n',lab_ck);

img_ck = reshape(data_img(ind_ck,:),I_high,I_wide);
figure; imshow(img_ck);

% answer = questdlg('Make sure you have checked the feature and label data correspondence instance. Proceed?',...
%                   'Feature and label data correspondence checking',...
%                   'Yes','No');
% switch answer
%     case 'Yes'
%         fprintf('Feature and label data correspondence checking done\n');
%     case 'No'
%         exit; 
% end

%% 
% f = msgbox('Congratulations! You have finished the dataset preparation!');



end
