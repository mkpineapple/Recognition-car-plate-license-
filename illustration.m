%% This file will illustrate the data
% JYI on 11/06/2018
% contributor QiQi and Ke Ma

%%
data_img = load('P607_50_Plates_Images.mat');
data_img = data_img.data_img;
data_feat = load('P607_50_Plates_Character_Images.mat');
data_feat = data_feat.data_feat;

data_cha = load('P607_50_Plate_Labels','data_cha');
data_cha = data_cha.data_cha;
data_lab = load('P607_50_Plate_Character_Labels.mat');
data_lab = data_lab.data_lab;

ck = randsample(50,1);
plate_img = reshape(data_img(ck,:),320,240);
figure; title('Plate image'); imshow(plate_img);

cha_ck = data_cha(ck);
fprintf('The plate reads as:\n');
cha_ck = cha_ck{1}

figure;
ch_ck = (ck-1)*7;
lab_cell = cell(1,7);
for i=1:7
    ch_ck = ch_ck+1;
    character_img = reshape(data_feat(ch_ck,:),40,20);

    subplot(1,7,i);
    imshow(character_img);

    ch_lab = data_lab(ch_ck,:);
    lab = vector_to_label(ch_lab,36);
    lab_cell{i} = lab;

end
