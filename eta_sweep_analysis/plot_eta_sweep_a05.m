%% ========================================================================
%  FedPDA η_g Sweep 비교 분석 — Dirichlet α=0.5 (Moderate Non-IID)
%  ========================================================================
%  Fig 1: Accuracy vs Round (η_g = 0.1, 0.3, 0.5 비교)
%  Fig 2: Accuracy vs Simulation Hours
%  Fig 3: Loss vs Round
%  Fig 4: Staleness Over Time (3개 η_g 비교)
%  Fig 5: Per-Plane Contributions (Grouped Bar, 3개 η_g)
%  Fig 6: Summary Table
%  ========================================================================

clear; clc; close all;

%% ========================= 경로 설정 ===================================
DATA_DIR = './data';
FIG_DIR  = './figures_a05';
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

%% ========================= 색상/스타일 ==================================
C.eta01 = [0.906, 0.298, 0.235];   % 빨강 (η_g=0.1, 보존 90%)
C.eta03 = [0.945, 0.600, 0.090];   % 주황 (η_g=0.3, 보존 70%)
C.eta05 = [0.580, 0.404, 0.741];   % 보라 (η_g=0.5, 보존 50%)

M.eta01 = 'o'; M.eta03 = 's'; M.eta05 = '^';

etas = {'eta01','eta03','eta05'};
eta_labels = {'\eta_g=0.1 (90%)', '\eta_g=0.3 (70%)', '\eta_g=0.5 (50%)'};
eta_vals = [0.1, 0.3, 0.5];

set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);

%% ========================= 데이터 로드 ==================================
merged = readtable(fullfile(DATA_DIR, 'merged_accuracy_a0.5.csv'), 'VariableNamingRule','preserve');
staleness = readtable(fullfile(DATA_DIR, 'merged_staleness_a0.5.csv'), 'VariableNamingRule','preserve');

%% ========================================================================
%  Figure 1 — Accuracy vs Aggregation Round
%% ========================================================================
fprintf('[Fig 1] Accuracy vs Round (alpha=0.5)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

for i = 1:3
    e = etas{i};
    col_name = sprintf('acc_eta%g', eta_vals(i));
    mk = max(1, floor(height(merged)/20));
    plot(merged.round, merged.(col_name), '-', ...
        'Color',C.(e), 'LineWidth',1.5, ...
        'Marker',M.(e), 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(merged), ...
        'MarkerFaceColor',C.(e), ...
        'DisplayName',eta_labels{i});
end

xlabel('Aggregation Round'); ylabel('Accuracy (%)');
title('FedPDA: Accuracy vs. Round (\alpha=0.5, \eta_g sweep)');
legend('Location','southeast'); ylim([0 90]);
exportgraphics(gcf, fullfile(FIG_DIR,'fig1_acc_vs_round.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 2 — Accuracy vs Simulation Hours
%% ========================================================================
fprintf('[Fig 2] Accuracy vs Hours (alpha=0.5)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

for i = 1:3
    e = etas{i};
    col_name = sprintf('acc_eta%g', eta_vals(i));
    mk = max(1, floor(height(merged)/20));
    plot(merged.sim_hours, merged.(col_name), '-', ...
        'Color',C.(e), 'LineWidth',1.5, ...
        'Marker',M.(e), 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(merged), ...
        'MarkerFaceColor',C.(e), ...
        'DisplayName',eta_labels{i});
end

xlabel('Simulation Time (hours)'); ylabel('Accuracy (%)');
title('FedPDA: Accuracy vs. Simulation Time (\alpha=0.5, \eta_g sweep)');
legend('Location','southeast'); ylim([0 90]);
exportgraphics(gcf, fullfile(FIG_DIR,'fig2_acc_vs_hours.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 3 — Loss vs Round
%% ========================================================================
fprintf('[Fig 3] Loss vs Round (alpha=0.5)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

for i = 1:3
    e = etas{i};
    col_name = sprintf('loss_eta%g', eta_vals(i));
    mk = max(1, floor(height(merged)/20));
    plot(merged.round, merged.(col_name), '-', ...
        'Color',C.(e), 'LineWidth',1.5, ...
        'Marker',M.(e), 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(merged), ...
        'MarkerFaceColor',C.(e), ...
        'DisplayName',eta_labels{i});
end

xlabel('Aggregation Round'); ylabel('Loss');
title('FedPDA: Loss vs. Round (\alpha=0.5, \eta_g sweep)');
legend('Location','northeast');
exportgraphics(gcf, fullfile(FIG_DIR,'fig3_loss_vs_round.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 4 — Staleness Over Time (3개 η_g 비교)
%% ========================================================================
fprintf('[Fig 4] Staleness Over Time (alpha=0.5)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

for i = 1:3
    e = etas{i};
    col_name = sprintf('mean_eta%g', eta_vals(i));
    if height(staleness) > 20
        sm = movmean(staleness.(col_name), 50);
    else
        sm = staleness.(col_name);
    end
    plot(staleness.sim_hours, sm, '-', ...
        'Color',C.(e), 'LineWidth',2, ...
        'DisplayName',eta_labels{i});
end

xlabel('Simulation Time (hours)'); ylabel('Mean Staleness (\tau)');
title('FedPDA: Staleness Over Time (\alpha=0.5, \eta_g sweep)');
legend('Location','northwest');
exportgraphics(gcf, fullfile(FIG_DIR,'fig4_staleness.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 5 — Per-Plane Contributions (Grouped Bar)
%% ========================================================================
fprintf('[Fig 5] Plane Contributions (alpha=0.5)\n');
figure('Position',[100 100 1000 500]); hold on; grid on; box on;

planes = (1:17)';
n_s = 3;
bw = 0.8 / n_s;
bar_mat = zeros(17, n_s);

plane_files = {
    fullfile(DATA_DIR, 'a0.5_eta0.1_plane_contributions.csv')
    fullfile(DATA_DIR, 'a0.5_eta0.3_plane_contributions.csv')
    fullfile(DATA_DIR, 'a0.5_eta0.5_plane_contributions.csv')
};

for i = 1:n_s
    if ~isfile(plane_files{i}), continue; end
    T = readtable(plane_files{i}, 'VariableNamingRule','preserve');
    total = sum(T.contributions);
    if total == 0, continue; end
    bar_mat(:,i) = T.contributions / total * 100;
end

for i = 1:n_s
    e = etas{i};
    offset = (i - (n_s+1)/2) * bw;
    bar(planes + offset, bar_mat(:,i), bw, ...
        'FaceColor',C.(e), 'FaceAlpha',0.85, ...
        'EdgeColor','w', 'DisplayName',eta_labels{i});
end

yline(100/17, '--', 'Color',[0.5 0.5 0.5], 'LineWidth',1, ...
    'DisplayName','Uniform (5.9%)');

xlabel('Orbital Plane ID'); ylabel('Contribution Share (%)');
title('FedPDA: Per-Plane Contribution (\alpha=0.5, \eta_g sweep)');
xticks(1:17); legend('Location','northeast');
exportgraphics(gcf, fullfile(FIG_DIR,'fig5_plane_contributions.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 6 — Summary Table
%% ========================================================================
fprintf('[Fig 6] Summary Table (alpha=0.5)\n');
figure('Position',[100 100 900 420]); axis off;

headers = {'Metric','\eta_g=0.1 (90%)','\eta_g=0.3 (70%)','\eta_g=0.5 (50%)'};
rows = {
    'Best Accuracy (%)',    '76.13',   '80.97',   '82.01'
    'Final Accuracy (%)',   '76.01',   '80.82',   '81.71'
    'Late-stage Mean (%)',  '74.34',   '79.80',   '80.85'
    'Late-stage Std (%)',   '1.15',    '0.82',    '0.73'
    'Time to 70% (h)',      '94.3',    '52.6',    '41.2'
    'Time to 80% (h)',      'N/A',     '138.1',   '108.1'
    'Total Rounds',         '365',     '365',     '365'
    'Avg Staleness',        '3.53',    '3.53',    '3.53'
};

nr = size(rows,1);
nc = 4;
cw = [0.28 0.22 0.22 0.22];
rh = 0.08;
x0 = 0.04; y0 = 0.82;

hdr_bg = {[0.84 0.85 0.87], [0.98 0.86 0.85], [1.0 0.93 0.80], [0.87 0.82 0.93]};

for j = 1:nc
    x = x0 + sum(cw(1:j-1));
    annotation('textbox',[x y0 cw(j) rh], ...
        'String',headers{j},'FontWeight','bold','FontSize',10, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'BackgroundColor',hdr_bg{j},'EdgeColor',[0.7 0.7 0.7],'Margin',2);
end

for i = 1:nr
    y = y0 - i*rh;
    for j = 1:nc
        x = x0 + sum(cw(1:j-1));
        str = rows{i,j};
        if j == 1, fw = 'bold'; ha = 'left';
        else,      fw = 'normal'; ha = 'center'; end

        bg = [1 1 1];
        % Best accuracy → η_g=0.5
        if i==1 && j==4, bg=[0.82 0.76 0.90]; fw='bold'; end
        % Best final → η_g=0.5
        if i==2 && j==4, bg=[0.82 0.76 0.90]; fw='bold'; end
        % Best 70% time → η_g=0.5
        if i==5 && j==4, bg=[0.82 0.76 0.90]; fw='bold'; end
        % Lowest std → η_g=0.5
        if i==4 && j==4, bg=[0.85 0.93 0.85]; fw='bold'; end
        % N/A → 회색
        if strcmp(str, 'N/A'), bg=[0.93 0.93 0.93]; end

        annotation('textbox',[x y cw(j) rh], ...
            'String',str,'FontWeight',fw,'FontSize',9.5, ...
            'HorizontalAlignment',ha,'VerticalAlignment','middle', ...
            'BackgroundColor',bg,'EdgeColor',[0.85 0.85 0.85],'Margin',2);
    end
end

title('FedPDA \eta_g Sweep Summary — \alpha=0.5 (Moderate Non-IID)','FontSize',13,'FontWeight','bold');
exportgraphics(gcf, fullfile(FIG_DIR,'fig6_summary_table.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
fprintf('\n==============================\n');
fprintf(' 6개 그래프 생성 완료 (alpha=0.5 eta_g sweep)\n');
fprintf(' 저장 위치: %s/\n', FIG_DIR);
fprintf('==============================\n');
