close all; clear all; clc;

end_time = 230;
ini_iterative = 0;
iterative = 450;
alpha = 0.5;

% Cost coefficients
% Q = 1;
% R = 1;

% Scaling 
xmin = [0, 0, 0.0001, 0, 0];
xmax = [230, 150, 25, 100, 110000];
umin = 10;
umax = 240;

%%

ic = zeros(460, iterative);
sc = zeros(1, iterative);
tc = zeros(1, iterative);

for k = 1:iterative
    disp(k)
    state = load(strcat('PL_state',num2str(k-1),'.txt'), '-ascii');
    input = load(strcat('PL_input',num2str(k-1),'.txt'), '-ascii'); input = input(:,1:end-1);
    reward = load(strcat('PL_reward',num2str(k-1),'.txt'), '-ascii');
    ic(:,k) = reward(1:end-1);
    sc(1,k) = reward(end);
    tc(1,k) = sum(ic(:,k)) + sc(1,k);

    local_input = input(1,:) + 5/3*input(2,:); 

    local_state(1,:) = state(1,:);
    local_state(2,:) = sum(state(2:5,:));
    local_state(3,:) = state(7,:);
    local_state(4,:) = state(8,:);
    local_state(5,:) = state(9,:);

    time = state(1,:);
    time_p = time(1:end-1);
    
    figure(1)
    subplot(2,2,1); hold on; plot(time, state(2,:), 'LineWidth', 2); title('A_0'); 
    subplot(2,2,2); hold on; plot(time, state(3,:), 'LineWidth', 2); title('A_1');
    subplot(2,2,3); hold on; plot(time, state(4,:), 'LineWidth', 2); title('A_3'); 
    subplot(2,2,4); hold on; plot(time, state(5,:), 'LineWidth', 2); title('A_4');

    figure(2)
    subplot(2,2,1); hold on; plot(time, state(6,:), 'LineWidth', 2); title('X');
    subplot(2,2,2); hold on; plot(time, state(7,:), 'LineWidth', 2); title('S');   
    subplot(2,2,3); hold on; plot(time, state(8,:), 'LineWidth', 2); title('P'); xlabel('Time (h)')   
    subplot(2,2,4); hold on; plot(time, state(9,:), 'LineWidth', 2); title('V'); xlabel('Time (h)')   

    figure(3)
    subplot(2,2,1); hold on; 
    plot(time, state(10,:), 'LineWidth', 2); title('T'); 
    plot(time, 298*ones(length(state),1), 'r', 'LineWidth', 2);  
    subplot(2,2,2); hold on; 
    plot(time, -log10(state(11,:)), 'LineWidth', 2); title('pH'); 
    plot(time, 6.5*ones(length(state),1), 'r', 'LineWidth', 2); 
    subplot(2,2,3); hold on; plot(time, state(12,:), 'LineWidth', 2); title('n_0');  
    subplot(2,2,4); hold on; plot(time, state(22,:), 'LineWidth', 2); title('n_m');  

    figure(4) 
    subplot(2,2,1); hold on; 
    plot(time, state(23,:), 'LineWidth', 2); title('DO2'); 
    plot(time, 2.2*ones(length(state),1),'r', 'LineWidth', 2); plot(time, 6.6*ones(length(state),1),'r', 'LineWidth', 2);  
    subplot(2,2,2); hold on; 
    plot(time, state(24,:), 'LineWidth', 2); title('DCO2'); 
    plot(time, 0.035*ones(length(state),1),'r', 'LineWidth', 2);  
    subplot(2,2,3); hold on; plot(time, state(25,:), 'LineWidth', 2); title('Viscosity');  
    subplot(2,2,4); hold on; plot(time, state(26,:), 'LineWidth', 2); title('PAA'); 
    plot(time, 200*ones(length(state),1),'r', 'LineWidth', 2); plot(time, 2000*ones(length(state),1),'r', 'LineWidth', 2);

    figure(5)
    subplot(2,2,1); hold on; plot(time, state(27,:), 'LineWidth', 2); title('O2'); 
    subplot(2,2,2); hold on; plot(time, state(28,:), 'LineWidth', 2); title('CO2'); 
    subplot(2,2,3); hold on; plot(time, state(29,:), 'LineWidth', 2); title('N'); 
    plot(time, 150*ones(length(state),1),'r', 'LineWidth', 2);

    figure(6)
    subplot(2,2,1); hold on; plot(time_p, input(1,:), 'LineWidth', 2); title('Fs'); 
    subplot(2,2,2); hold on; plot(time_p, input(2,:), 'LineWidth', 2); title('Foil');
    subplot(2,2,3); hold on; plot(time_p, input(3,:), 'LineWidth', 2); title('Fpaa'); 
    subplot(2,2,4); hold on; plot(time_p, input(6,:), 'LineWidth', 2); title('Fw');

    figure(7)
    subplot(2,2,1); hold on; plot(time_p, input(8,:), 'LineWidth', 2); title('Fc'); 
    subplot(2,2,2); hold on; plot(time_p, input(9,:), 'LineWidth', 2); title('Fh'); 
    subplot(2,2,3); hold on; plot(time_p, input(4,:), 'LineWidth', 2); title('Fa');
    subplot(2,2,4); hold on; plot(time_p, input(5,:), 'LineWidth', 2); title('Fb'); 


    figure(8)
    subplot(2,2,1); hold on; plot(time_p, input(7,:), 'LineWidth', 2); title('Fg');
    subplot(2,2,2); hold on; plot(time_p, input(10,:), 'LineWidth', 2); title('Pressure'); 
    subplot(2,2,3); hold on; plot(time_p, input(11,:), 'LineWidth', 2); title('NH3_shots');
    subplot(2,2,4); hold on; plot(time(120:end), state(7,120:end), 'LineWidth', 2); title('Partial S'); 

    if k < ini_iterative + 1
        cc = 'black';
    else
        cc = 'blue';
    end
    
    figure(9)
    local_state_patch = local_state; time_patch = time;
    local_state_patch(:,end+1) = NaN*ones(5,1); time_patch(:,end+1) = time(end);
    subplot(2,2,1); hold on; title('Biomass concentration'); 
    patch(time_patch, local_state_patch(2,:),'red','EdgeColor',cc,'FaceVertexAlphaData',alpha*ones(length(time_patch),1),'AlphaDataMapping','none','EdgeAlpha','interp','Linestyle',':')
    subplot(2,2,2); hold on; title('Substrate concentration');  
    patch(time_patch,local_state_patch(3,:),'red','EdgeColor',cc,'FaceVertexAlphaData',alpha*ones(length(time_patch),1),'AlphaDataMapping','none','EdgeAlpha','interp','Linestyle',':')
    subplot(2,2,3); hold on; title('Penicillin concentration');  
    patch(time_patch,local_state_patch(4,:),'red','EdgeColor',cc,'FaceVertexAlphaData',alpha*ones(length(time_patch),1),'AlphaDataMapping','none','EdgeAlpha','interp','Linestyle',':')
    subplot(2,2,4); hold on; title('Culture volume');
    patch(time_patch,local_state_patch(5,:),'red','EdgeColor',cc,'FaceVertexAlphaData',alpha*ones(length(time_patch),1),'AlphaDataMapping','none','EdgeAlpha','interp','Linestyle',':')

    figure(10)
    local_input_patch = local_input;
    local_input_patch(:,end+1) = NaN;
    hold on; title('Substrate feed rate')
    patch(time, local_input_patch,'red','EdgeColor',cc,'FaceVertexAlphaData',alpha*ones(length(time),1),'AlphaDataMapping','none','EdgeAlpha','interp','Linestyle',':')


end

figure(10); hold on;
plot(time_p, umax*ones(length(time_p),1),'--r'); plot(time_p, umin*ones(length(time_p),1),'--r');
xlabel('Time (h)'); ylabel('Value'); xlim([0, 230]);

figure(11); hold on; title('costs')
plot(sum(ic)); plot(sc); plot(tc);

%%
% 
% for k = ini_iterative + 1:iterative
%     
%     mpc_state = load(strcat('MPC_state',num2str(k-1),'.txt'), '-ascii');
%     mpc_input = load(strcat('MPC_input',num2str(k-1),'.txt'), '-ascii'); mpc_input = mpc_input(:,1:end-1);
%     mpc_flag = load(strcat('MPC_flag',num2str(k-1),'.txt'), '-ascii');
%     mpc_cost = load(strcat('MPC_cost',num2str(k-1),'.txt'), '-ascii');
%  
%     pe_state = load(strcat('PE_state',num2str(k-1),'.txt'), '-ascii');
%     pe_input = load(strcat('PE_input',num2str(k-1),'.txt'), '-ascii'); pe_input = pe_input(:,1:end-1);
%     pe_para = load(strcat('PE_para',num2str(k-1),'.txt'), '-ascii');
%     pe_flag = load(strcat('PE_flag',num2str(k-1),'.txt'), '-ascii');
%     pe_cost = load(strcat('PE_cost',num2str(k-1),'.txt'), '-ascii');
%     pe_death(k) = sum(pe_flag == 5)/30*100;
%     mpc_death(k) = sum(mpc_flag == 9)/450*100; %% 10 or 9 ?
%     
%     
%     figure(12)
%     subplot(2,2,1); hold on; plot(time_p, pe_flag); title('PE flag');
%     subplot(2,2,2); hold on; plot(time_p, pe_cost); title('PE cost');
%     subplot(2,2,3); hold on; plot(time_p, mpc_flag); title('MPC flag');
%     subplot(2,2,4); hold on; plot(time_p, mpc_cost); title('MPC cost');
%     
%     figure(13); hold on
%     plot(time_p, mpc_input); title('MPC input');
% 
% end
% 
% figure(13)
% plot(time_p, umax*ones(length(time_p),1),'--r'); plot(time_p, umin*ones(length(time_p),1),'--r');
% xlabel('Time (h)'); ylabel('Value'); xlim([0, 230]);
% 
% figure(14)
% subplot(2,1,1); hold on; plot(pe_death); title('PE death ratio');
% subplot(2,1,2); hold on; plot(mpc_death); title('MPC death ratio');

