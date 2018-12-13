function cha_v = label_to_vector(cha,N_class)
%% This file is to turn labels into one-hot vectors
% JYI on 11/06/2018
% contributor: QiQi, Ke Ma
cha_v = zeros(1,N_class);

switch cha
    case '0'
        cha_v(1) = 1;
    case '1'
        cha_v(2) = 1;
    case '2'
        cha_v(3) = 1;
    case '3'
        cha_v(4) = 1;
    case '4'
        cha_v(5) = 1;
    case '5'
        cha_v(6) = 1;
    case '6'
        cha_v(7) = 1;
    case '7'
        cha_v(8) = 1;
    case '8'
        cha_v(9) = 1;
    case '9'
        cha_v(10) = 1;
    case 'A'
        cha_v(11) = 1;
    case 'B'
        cha_v(12) = 1;
    case 'C'
        cha_v(13) = 1;
    case 'D'
        cha_v(14) = 1;
    case 'E'
        cha_v(15) = 1;
    case 'F'
        cha_v(16) = 1;
    case 'G'
        cha_v(17) = 1;
    case 'H'
        cha_v(18) = 1;
    case 'I'
        cha_v(19) = 1;
    case 'J'
        cha_v(20) = 1;
    case 'K'
        cha_v(21) = 1;
    case 'L'
        cha_v(22) = 1;
    case 'M'
        cha_v(23) = 1;
    case 'N'
        cha_v(24) = 1;
    case 'O'
        cha_v(25) = 1;
    case 'P'
        cha_v(26) = 1;
    case 'Q'
        cha_v(27) = 1;
    case 'R'
        cha_v(28) = 1;
    case 'S'
        cha_v(29) = 1;
    case 'T'
        cha_v(30) = 1;
    case 'U'
        cha_v(31) = 1;
    case 'V'
        cha_v(32) = 1;
    case 'W'
        cha_v(33) = 1;
    case 'X'
        cha_v(34) = 1;
    case 'Y'
        cha_v(35) = 1;
    case 'Z'
        cha_v(36) = 1;

end

end
