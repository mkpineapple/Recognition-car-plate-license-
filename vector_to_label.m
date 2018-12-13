function lab = vector_to_label(vec_lab,N_class)
%% This file is to turn one-hot vectors into labels
% JYI on 11/06/2018
% contributor: QiQi and Ke Ma

    lab_ind = find(vec_lab==1);
    switch lab_ind
        case 1
            lab = 0;
        case 2
            lab = 1;
        case 3
            lab = 2;
        case 4
            lab = 3;
        case 5
            lab = 4;
        case 6
            lab = 5;
        case 7
            lab = 6;
        case 8
            lab = 7;
        case 9
            lab = 8;
        case 10
            lab = 9;
        case 11
            lab = 'A';
        case 12
            lab = 'B';
        case 13
            lab = 'C';
        case 14
            lab = 'D';
        case 15
            lab = 'E';
        case 16
            lab = 'F';
        case 17
            lab = 'G';
        case 18
            lab = 'H';
        case 19
            lab = 'I';
        case 20
            lab = 'J';
        case 21
            lab = 'K';
        case 22
            lab = 'L';
        case 23
            lab = 'M';
        case 24
            lab = 'N';
        case 25
            lab = 'O';
        case 26
            lab = 'P';
        case 27
            lab = 'Q';
        case 28
            lab = 'R';
        case 29
            lab = 'S';
        case 30
            lab = 'T';
        case 31
            lab = 'U';
        case 32
            lab = 'V';
        case 33
            lab = 'W';
        case 34
            lab = 'X';
        case 35
            lab = 'Y';
        case N_class
            lab = 'Z';

    end

end
