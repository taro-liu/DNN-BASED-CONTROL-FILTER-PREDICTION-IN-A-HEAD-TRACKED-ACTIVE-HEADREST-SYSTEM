%% Control filters calculation

clc
clear all

addpath(genpath('D:\haowen_ANC\lib\'))
addpath(genpath('D:\haowen_ANC\lib\mimo_wiener\'))

filter_len = 256;
load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_pri_2_1_3_37.mat");
pri_path_all = h';
load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_sec_2_1_3_37.mat")
sec_path = h';
[len_pos, len_f] = size(h);
W_model = zeros(size(h));
[noise, fs] = audioread("D:\haowen_ANC\whitenoise.wav");
ref = noise(1:10*fs);
for i = 1:len_pos
    err = filter(pri_path_all(:,i), 1, ref);
    sec_path_true = sec_path(:,i);
    [w, ~] = mimo_wiener_anc(ref, err, 256, sec_path_true, 1e-4, ref, err, sec_path_true);
    W_model(i,:) = w;
end

save("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_w_2_1_3_37.mat", 'W_model');