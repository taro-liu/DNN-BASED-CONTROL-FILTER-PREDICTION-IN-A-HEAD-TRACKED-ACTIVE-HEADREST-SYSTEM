%% NR calculation
clc
clear all

addpath(genpath('D:\haowen_ANC\lib\'))
addpath(genpath('D:\haowen_ANC\lib\mimo_wiener\'))

filter_len = 256;
load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_pri_3.7_1.3_2_43.mat");
pri_path_true = h';

pri_path_model = importdata("D:\database\comsol\analysis\exp\2025_exp\new-exp\model_pri_3.7_1.3_2_43_bs4.txt");
pri_path_model = pri_path_model';


load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_sec_3.7_1.3_2_43.mat")
sec_path_true = h';


sec_path_model = importdata("D:\database\comsol\analysis\exp\2025_exp\new-exp\model_sec_3.7_1.3_2_43.txt");
sec_path_model = sec_path_model';

load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_sec_5_-5_0_-60.mat");
sec_path_nearest = h';
load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\sec_path_all.mat")
sec_path_initial = h(562,:)';

load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\control_filters_all.mat")
W = W(845,:)';
W1 = importdata("D:\database\comsol\analysis\exp\2025_exp\new-exp\model_W_3.7_1.3_2_43.txt");
W1 = W1';

W_label = importdata("D:\database\comsol\analysis\exp\2025_exp\new-exp\tra_interp_W_spatial_3.7_1.3_2_43.txt");
W_label = W_label';

load("D:\database\comsol\analysis\exp\2025_exp\new-exp\FIR\label_pri_5_-5_0_-60.mat");
pri_path_nearest = h';

[noise, fs] = audioread("D:\haowen_ANC\whitenoise.wav");
ref = noise(1:10*fs);
err_true = filter(pri_path_true, 1, ref);
err_model = filter(pri_path_model, 1, ref);
err_nearest = filter(pri_path_nearest, 1, ref);

[~, e_w_true] = mimo_wiener_anc(ref, err_true, 256, sec_path_true, 1e-4, ref, err_true, sec_path_true);
[~, e_path] = mimo_wiener_anc(ref, err_model, 256, sec_path_model, 1e-4, ref, err_true, sec_path_true);
[~, ~, e_w_model] = mimo_fxlms_liu(0, filter_len, ref, err_true, ref, err_true, sec_path_model, sec_path_true, sec_path_true, W1); % 调用模型预测的次级路径
[~, e_w_nearest_W] = mimo_fxlms_liu(0, filter_len, ref, err_true, ref, err_true, sec_path_model, sec_path_true, sec_path_true, W);
% [W_4, e_4, e_W_model] = mimo_fxlms_liu(0, filter_len, ref, err_true, ref, err_true, sec_path_model, sec_path_true, sec_path_true, W); % 调用模型预测的次级路径
nfft = 256;
tfb = [100, 2000];
fs = 5120;
noverlap = 128;

[~, spl_A_noise, ~, ~] = calc_spl_A(err_true, fs, tfb, nfft, noverlap);
[~, spl_A_w_label, ~, ~] = calc_spl_A(e_w_true, fs, tfb, nfft, noverlap);
[~, spl_A_w_model, ~, ~] = calc_spl_A(e_w_model, fs, tfb, nfft, noverlap);
[~, spl_A_path, ~, ~] = calc_spl_A(e_path, fs, tfb, nfft, noverlap);
[~, spl_A_w_nearest_W, ~, ~] = calc_spl_A(e_w_nearest_W, fs, tfb, nfft, noverlap);

nr_label = spl_A_noise - spl_A_w_label
nr_model = spl_A_noise - spl_A_w_model
nr_path = spl_A_noise - spl_A_path
nr_nearest_W = spl_A_noise - spl_A_w_nearest_W

% 
% [~, e_w_true] = mimo_wiener_anc(ref, err_true, 256, sec_path_true, 1e-4, ref, err_true, sec_path_true);
% [~, e_w_model] = mimo_wiener_anc(ref, err_true, 256, sec_path_model, 1e-4, ref, err_true, sec_path_true);
% [~, e_w_nearest] = mimo_wiener_anc(ref, err_true, 256, sec_path_nearest, 1e-4, ref, err_true, sec_path_true);
% [~, e_w_nearest_W] = mimo_wiener_anc(ref, err_nearest, 256, sec_path_nearest, 1e-4, ref, err_true, sec_path_true);
% [W_4, e_4, e_W_model] = mimo_fxlms_liu(0, filter_len, ref, err_true, ref, err_true, sec_path_model, sec_path_true, sec_path_true, W); % 调用模型预测的次级路径
% 
% 
% 
% 
% % [~, e_w_true] = mimo_wiener_anc(ref, err_true, 256, sec_path_true, 1e-4, ref, err_true, sec_path_true);
% % [~, e_w_model_p_true] = mimo_wiener_anc(ref, err_true, 256, sec_path_model, 1e-4, ref, err_true, sec_path_true);
% % [~, e_w_model] = mimo_wiener_anc(ref, err_model, 256, sec_path_model, 1e-4, ref, err_true, sec_path_true);
% % [W_3, e_4, e_fxlms_model_p_true] = mimo_fxlms_liu(0.01, filter_len, ref, err_true, ref, err_true, sec_path_model, sec_path_true, sec_path_true);
% % [W_3, e_4, e_fxlms_model] = mimo_fxlms_liu(0.01, filter_len, ref, err_model, ref, err_true, sec_path_model, sec_path_true, sec_path_true);
% nfft = 256;
% tfb = [100, 2000];
% fs = 5120;
% noverlap = 128;
% 
% [~, spl_A_noise, ~, ~] = calc_spl_A(err_true, fs, tfb, nfft, noverlap);
% [~, spl_A_w_label, ~, ~] = calc_spl_A(e_w_true, fs, tfb, nfft, noverlap);
% [~, spl_A_w_model, ~, ~] = calc_spl_A(e_w_model, fs, tfb, nfft, noverlap);
% [~, spl_A_W_model, ~, ~] = calc_spl_A(e_W_model, fs, tfb, nfft, noverlap);
% [~, spl_A_w_nearest, ~, ~] = calc_spl_A(e_w_nearest, fs, tfb, nfft, noverlap);
% [~, spl_A_w_nearest_W, ~, ~] = calc_spl_A(e_w_nearest_W, fs, tfb, nfft, noverlap);
% 
% % [~, spl_A_w_label, ~, ~] = calc_spl_A(e_w_true, fs, tfb, nfft, noverlap);
% % [~, spl_A_noise, ~, ~] = calc_spl_A(err_true, fs, tfb, nfft, noverlap);
% % [~, spl_A_w_model_p_true, ~, ~] = calc_spl_A(e_w_model_p_true, fs, tfb, nfft, noverlap);
% % [~, spl_A_w_model, ~, ~] = calc_spl_A(e_w_model, fs, tfb, nfft, noverlap);
% % [~, spl_A_f_model_p_true, ~, ~] = calc_spl_A(e_fxlms_model_p_true, fs, tfb, nfft, noverlap);
% % [~, spl_A_f_model, ~, ~] = calc_spl_A(e_fxlms_model, fs, tfb, nfft, noverlap);
% % 
% % nr_label = spl_A_noise - spl_A_w_label
% % nr_w_model_p_true = spl_A_noise - spl_A_w_model_p_true
% % nr_w_model = spl_A_noise - spl_A_w_model
% % nr_f_model_p_true = spl_A_noise - spl_A_f_model_p_true
% % nr_f_model = spl_A_noise - spl_A_f_model
% 
% 
% nr_label = spl_A_noise - spl_A_w_label
% nr_model = spl_A_noise - spl_A_w_model
% nr_W_model = spl_A_noise - spl_A_W_model
% nr_nearest = spl_A_noise - spl_A_w_nearest
% nr_nearest_W = spl_A_noise - spl_A_w_nearest_W