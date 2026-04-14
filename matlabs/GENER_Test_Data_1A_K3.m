%% DOA数据生成参数设置
clear; clc;
% 阵列参数
N = 16;               % 阵元数
lambda = 1;           % 波长（归一化）
d = lambda/2;         % 阵元间距

% 数据包参数
T = 20;              % 每包快拍数
num_packets = 5;   % 数据包数量
SNR_range = -5;  % 确定信噪比(dB)

% 网格定义
theta_grid = -60:1:60;  % 1°间隔的网格点
SOURCE_interval = 10;
SOURCE_K = 3;
load('Dict.mat');
rng(10086)


% %% 执行数据生成（示例）
% dataset = generate_doa_dataset(N, T, num_packets, SNR_range, theta_grid);
% 


%% 主数据生成函数
% function dataset = generate_doa_dataset(N, T, num_packets, SNR_range, theta_grid)
% 初始化数据集
% dataset = struct('snapshots', {}, 'angles', {});

all_angles = zeros(num_packets,SOURCE_K,T);
all_snapshots = zeros(num_packets,2,T,N);
all_S_est = zeros(num_packets,2,T,length(theta_grid));
all_Rx = zeros(num_packets,T,2,N,N);
all_sbl = zeros(num_packets*T, SOURCE_K);

L=length(theta_grid);
A=exp(1i*pi*(0:N-1)'*sind(theta_grid));
H=zeros(N*N,L);
for i=1:N
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta_grid)));
    H((i-1)*N+1:i*N,:)=fhi;   %A的拼接
end
start_index = 1;

% 遍历所有数据包
for p = 1:num_packets
    % 生成当前数据包
    [Rx_tmp, snapshots, angles, angles_sbl] = generate_packet(N, T, theta_grid, SOURCE_interval, start_index, SNR_range, SOURCE_K, AA);
    all_sbl((p-1)*T+1:p*T,:,:) = angles_sbl';
    % 存入数据集
    % dataset(end+1).snapshots = snapshots;
    % dataset(end).angles = angles;
    all_angles(p,:,:) = angles;
    % all_snapshots(p,:,:,:) = snapshots;
    % all_S_est(p,:,:,:) = S_est;
    all_Rx(p,:,:,:,:) = Rx_tmp;

    start_index = start_index+T-1;
end

% 覆盖性验证
% all_angles = verify_coverage(dataset, theta_grid);
% end

filename = 'D:\files\reafiles\Iterative_DOA\denoiser_deep_unfolding\RUST_doa\modules\data\test\Result_SBL_GridAligned_Smooth_test_uniform_running10deg_20_5_m5dB_1A_K3.h5';
h5create(filename,'/SBL', size(all_sbl));
h5write(filename, '/SBL', all_sbl);

% %% 数据保存（可选）
% save('D:\files\reafiles\Iterative_DOA\denoiser_deep_unfolding\RUST_doa\modules\data\test\DOA_Dataset_GridAligned_Smooth_test_uniform_running5deg_20_5_m5dB_1A_K3.mat', 'all_angles',"all_Rx");
% disp('数据集已保存为DOA_Dataset_GridAligned_Smooth.mat');

%% 单个数据包生成函数
function [Rx_tmp, snapshots, angles, angles_sbl] = generate_packet(N, T, theta_grid, interval, start_index, SNR, SOURCE_K, AA)
    % 初始化存储
    snapshots = zeros(N, T);  % 复数接收信号
    angles = zeros(SOURCE_K, T);     % 真实角度
    Rx_tmp = zeros(T,2,N,N);  %协方差矩阵
    angles_sbl = zeros(3, T);
    
    % 固定角度间隔
    % interval = 10;  % 两个信号源的角度间隔
    
    % 随机生成初始角度（确保两个信号源的初始角度在网格范围内）
    theta1_start_idx = start_index;  % 信号源1的起点索引
    theta1_start_idx = max(1, theta1_start_idx);  % 确保起点索引 ≥1
    theta1_start = theta_grid(theta1_start_idx);  % 信号源1的起点角度
    theta2_start = theta1_start + interval;  % 信号源2的起点角度
    theta3_start = theta2_start + interval;  % 信号源3的起点角度
    
    % 生成连续轨迹（每个快拍移动1°）
    for t = 1:T
        % 计算当前角度
        theta1 = theta1_start + (t - 1);  % 信号源1的角度
        theta2 = theta2_start + (t - 1);  % 信号源2的角度
        theta3 = theta3_start + (t - 1);  % 信号源3的角度
        
        % 记录当前角度
        angles(:, t) = [theta1; theta2;theta3];
    end
    
    % 信号合成
    for t = 1:T
        % 当前角度
        current_theta1 = angles(1, t);
        current_theta2 = angles(2, t);
        current_theta3 = angles(3, t);
        
        % 生成导向矢量
        a1 = exp(-1j * pi * (0:N-1)' * sind(current_theta1));
        a2 = exp(-1j * pi * (0:N-1)' * sind(current_theta2));
        a3 = exp(-1j * pi * (0:N-1)' * sind(current_theta3));
        
        % 生成信号源（恒定幅度，随机相位）
        s1 = exp(1j * 2*pi*rand());
        s2 = exp(1j * 2*pi*rand());
        s3 = exp(1j * 2*pi*rand());
        
        % 合成接收信号
        x = a1 * s1 + a2 * s2 + a3 * s3;
        
        % 添加高斯白噪声
        % SNR = SNR_range(1) + (SNR_range(2) - SNR_range(1)) * rand();
        noise_power = 10^(-SNR / 10);
        noise = sqrt(noise_power / 2) * (randn(N, 1) + 1j * randn(N, 1));
        x = x + noise;

        [~,pred_angles,~] = SBAC_K3(x,AA,100,0);
        angles_sbl(:,t) = sort(pred_angles);
        
        snapshots(:, t) = x;
        Rx = x*x';
        Rx_tmp(t,1,:,:) = real(Rx);
        Rx_tmp(t,2,:,:) = imag(Rx);
    end
end
