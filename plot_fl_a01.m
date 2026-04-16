%% ========================================================================
%  LEO 위성 연합학습 비교 실험 시각화 — Dirichlet α=0.1 (Severe Non-IID)
%  ========================================================================
%  Fig 1: Accuracy vs Round (5전략)
%  Fig 2: Accuracy vs Simulation Hours (FedSpace + FedOrbit + FedPDA)
%  Fig 3: Plane별 집계 기여 (Grouped Bar, 5전략)
%  Fig 4: 버퍼 Plane 다양성 (FedBuff vs FedSpace vs FedPDA)
%  Fig 5: 통신 기회 활용률 (Stacked Bar, 5전략) — α와 무관 (동일)
%  Fig 6: 위성 참여 공정성 (Lorenz Curve + Gini, 5전략)
%  Fig 7: FedSpace 동적 Threshold
%  Fig 8: 종합 비교 테이블 (5전략, α=0.1)
%  Fig 9: FedOrbit Staleness & Plane
%  Fig 10: FedPDA Staleness & Plane
%  Fig 11: α=0.5 vs α=0.1 비교 테이블 (NEW)
%  ========================================================================

clear; clc; close all;

%% ========================= 경로 설정 ===================================
DATA_DIR = './plot_dir_v4 - Dirichlet 0.1';   % ← α=0.1 CSV 파일 폴더
FIG_DIR  = './figure_dir_v4 - Dirichlet 0.1__';    % ← 출력 폴더
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

P = 'a01_';  % 파일 접두사

%% ========================= 색상/스타일 ==================================
C.FedAsync  = [0.906, 0.298, 0.235];   % 빨강
C.FedBuff   = [0.204, 0.596, 0.859];   % 파랑
C.FedSpace  = [0.180, 0.800, 0.443];   % 초록
C.FedOrbit  = [0.945, 0.600, 0.090];   % 주황
C.FedPDA    = [0.580, 0.404, 0.741];   % 보라

M.FedAsync = 'o'; M.FedBuff = 's'; M.FedSpace = '^'; M.FedOrbit = 'd'; M.FedPDA = 'p';

strats = {'FedAsync','FedBuff','FedSpace','FedOrbit','FedPDA'};

set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);

%% ========================================================================
%  Figure 1 — Accuracy vs Aggregation Round (5전략, α=0.1)
%% ========================================================================
fprintf('[Fig 1] Accuracy vs Round (alpha=0.1)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

perf_files = {
    fullfile(DATA_DIR, [P 'clean_perf_fedasync.csv'])
    fullfile(DATA_DIR, [P 'clean_perf_fedbuff.csv'])
    fullfile(DATA_DIR, [P 'clean_perf_fedspace.csv'])
    fullfile(DATA_DIR, [P 'clean_perf_fedorbit.csv'])
    fullfile(DATA_DIR, [P 'clean_perf_fedpda.csv'])
};

for i = 1:5
    s = strats{i};
    f = perf_files{i};
    if ~isfile(f), warning('Missing: %s',f); continue; end
    T = readtable(f);
    mk = max(1, floor(height(T)/20));
    plot(T.agg_round, T.accuracy, '-', ...
        'Color',C.(s), 'LineWidth',1.5, ...
        'Marker',M.(s), 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(T), ...
        'MarkerFaceColor',C.(s), ...
        'DisplayName',s);
end

xlabel('Aggregation Round'); ylabel('Accuracy (%)');
title('Global Model Accuracy vs. Aggregation Rounds (\alpha=0.1)');
legend('Location','southeast'); ylim([0 85]);
exportgraphics(gcf, fullfile(FIG_DIR,'fig1_accuracy_vs_round_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 2 — Accuracy vs Simulation Hours (FedSpace + FedOrbit + FedPDA)
%% ========================================================================
fprintf('[Fig 2] Accuracy vs Hours (alpha=0.1)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

% FedSpace
f = fullfile(DATA_DIR, [P 'fedspace_accuracy.csv']);
if isfile(f)
    T = readtable(f);
    mk = max(1, floor(height(T)/20));
    plot(T.sim_hours, T.accuracy, '-', ...
        'Color',C.FedSpace, 'LineWidth',1.5, ...
        'Marker','^', 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(T), ...
        'MarkerFaceColor',C.FedSpace, ...
        'DisplayName','FedSpace');
end

% FedOrbit
f = fullfile(DATA_DIR, [P 'fedorbit_accuracy_hours.csv']);
if isfile(f)
    T = readtable(f);
    mk = max(1, floor(height(T)/20));
    plot(T.sim_hours, T.accuracy, '-', ...
        'Color',C.FedOrbit, 'LineWidth',1.5, ...
        'Marker','d', 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(T), ...
        'MarkerFaceColor',C.FedOrbit, ...
        'DisplayName','FedOrbit');
end

% FedPDA
f = fullfile(DATA_DIR, [P 'fedpda_accuracy.csv']);
if isfile(f)
    T = readtable(f);
    mk = max(1, floor(height(T)/20));
    plot(T.sim_hours, T.accuracy, '-', ...
        'Color',C.FedPDA, 'LineWidth',1.5, ...
        'Marker','p', 'MarkerSize',5, ...
        'MarkerIndices',1:mk:height(T), ...
        'MarkerFaceColor',C.FedPDA, ...
        'DisplayName','FedPDA');
end

xlabel('Simulation Time (hours)'); ylabel('Accuracy (%)');
title('Global Model Accuracy vs. Simulation Time (\alpha=0.1)');
legend('Location','southeast'); ylim([0 85]);
exportgraphics(gcf, fullfile(FIG_DIR,'fig2_accuracy_vs_hours_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 3 — Per-Plane Aggregation Contribution (Grouped Bar, 5전략)
%% ========================================================================
fprintf('[Fig 3] Plane Contributions (alpha=0.1)\n');
figure('Position',[100 100 1100 500]); hold on; grid on; box on;

plane_files = {
    fullfile(DATA_DIR, [P 'fedasync_plane_contributions_corrected.csv'])
    fullfile(DATA_DIR, [P 'clean_plane_fedbuff.csv'])
    fullfile(DATA_DIR, [P 'fedspace_plane_contributions_corrected.csv'])
    fullfile(DATA_DIR, [P 'fedorbit_plane_contributions_corrected.csv'])
    fullfile(DATA_DIR, [P 'fedpda_plane_contributions_corrected.csv'])
};

planes = (1:17)';
n_s = 5;
bw = 0.8 / n_s;
bar_mat = zeros(17, n_s);

for i = 1:n_s
    if ~isfile(plane_files{i}), continue; end
    T = readtable(plane_files{i});
    total = sum(T.contributions);
    if total == 0, continue; end
    bar_mat(:,i) = T.contributions / total * 100;
end

for i = 1:n_s
    offset = (i - (n_s+1)/2) * bw;
    bar(planes + offset, bar_mat(:,i), bw, ...
        'FaceColor',C.(strats{i}), 'FaceAlpha',0.85, ...
        'EdgeColor','w', 'DisplayName',strats{i});
end

yline(100/17, '--', 'Color',[0.5 0.5 0.5], 'LineWidth',1, ...
    'DisplayName','Uniform (5.9%)');

xlabel('Orbital Plane ID'); ylabel('Contribution Share (%)');
title('Per-Plane Aggregation Contribution (\alpha=0.1)');
xticks(1:17); legend('Location','northeast','NumColumns',2);
exportgraphics(gcf, fullfile(FIG_DIR,'fig3_plane_contributions_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 4 — Buffer Plane Diversity (FedBuff vs FedSpace vs FedPDA)
%% ========================================================================
fprintf('[Fig 4] Buffer Diversity (alpha=0.1)\n');
figure('Position',[100 100 1200 500]);

buf_files = {
    fullfile(DATA_DIR, [P 'clean_bufdiv_fedbuff.csv']),    'FedBuff'
    fullfile(DATA_DIR, [P 'fedspace_buffer_diversity.csv']), 'FedSpace'
    fullfile(DATA_DIR, [P 'fedpda_buffer_diversity.csv']),   'FedPDA'
};

for idx = 1:3
    fname = buf_files{idx,1};
    sname = buf_files{idx,2};
    if ~isfile(fname), continue; end
    T = readtable(fname);

    subplot(1,3,idx); hold on; grid on; box on;

    uniq_d = unique(T.num_unique_planes);
    cnt = arrayfun(@(v) sum(T.num_unique_planes==v), uniq_d);
    pct = cnt / height(T) * 100;

    bar(uniq_d, pct, 'FaceColor',C.(sname), 'FaceAlpha',0.85, 'EdgeColor','w');

    for j = 1:numel(pct)
        if pct(j) > 3
            text(uniq_d(j), pct(j)+1.5, sprintf('%.1f%%',pct(j)), ...
                'HorizontalAlignment','center','FontSize',9);
        end
    end

    avg_d = mean(T.num_unique_planes);
    avg_k = mean(T.k);
    xline(avg_d, 'r--', 'LineWidth',1.5);
    text(avg_d+0.15, max(pct)*0.9, sprintf('Mean=%.1f',avg_d), ...
        'Color','r','FontWeight','bold','FontSize',10);

    xlabel('Number of Unique Planes in Buffer');
    ylabel('Frequency (%)');
    title(sprintf('%s  (avg K=%.1f)', sname, avg_k));
    xlim([0.5, max(uniq_d)+0.5]);
end

sgtitle('Buffer Plane Diversity per Flush (\alpha=0.1)','FontSize',14);
exportgraphics(gcf, fullfile(FIG_DIR,'fig4_buffer_diversity_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 5 — Communication Opportunity Utilization (동일 — 궤도 역학 결정)
%% ========================================================================
fprintf('[Fig 5] Communication Utilization (same as alpha=0.5)\n');
figure('Position',[100 100 900 500]); hold on; grid on; box on;

comm = [5880,  587,  5282;    % FedAsync
        5880, 2146,  3709;    % FedBuff
        5880, 2131,  3729;    % FedSpace
        4253,  220,  3591;    % FedOrbit
        5880, 2131,  3729];   % FedPDA

up_pct  = comm(:,2)./comm(:,1)*100;
dl_pct  = comm(:,3)./comm(:,1)*100;
sk_pct  = 100 - up_pct - dl_pct;

b = bar(1:5, [up_pct, dl_pct, sk_pct], 'stacked');
b(1).FaceColor = [0.180 0.800 0.443]; b(1).DisplayName = 'Upload (Model Contribution)';
b(2).FaceColor = [0.204 0.596 0.859]; b(2).FaceAlpha = 0.7; b(2).DisplayName = 'Download (Global Sync)';
b(3).FaceColor = [0.584 0.647 0.651]; b(3).FaceAlpha = 0.6; b(3).DisplayName = 'Skip (Up-to-date)';

for i = 1:5
    text(i, up_pct(i)/2, sprintf('%.1f%%',up_pct(i)), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',11,'FontWeight','bold','Color','w');
end

set(gca,'XTick',1:5,'XTickLabel',strats);
xlabel('Strategy'); ylabel('Share of GS Contacts (%)');
title('Communication Opportunity Utilization (invariant to \alpha)');
legend('Location','northeast'); ylim([0 105]);
exportgraphics(gcf, fullfile(FIG_DIR,'fig5_comm_utilization_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 6 — Lorenz Curve (Satellite Participation Fairness, 5전략)
%% ========================================================================
fprintf('[Fig 6] Lorenz Curve (alpha=0.1)\n');
figure('Position',[100 100 700 700]); hold on; grid on; box on;

plot([0 1],[0 1],'k--','LineWidth',1,'DisplayName','Perfect Equality');

sat_files = {
    fullfile(DATA_DIR, [P 'fedasync_sat_contributions.csv']),  'FedAsync'
    fullfile(DATA_DIR, [P 'clean_sat_fedbuff.csv']),           'FedBuff'
    fullfile(DATA_DIR, [P 'fedspace_sat_contributions.csv']),  'FedSpace'
    fullfile(DATA_DIR, [P 'fedorbit_sat_contributions.csv']),  'FedOrbit'
    fullfile(DATA_DIR, [P 'fedpda_sat_contributions.csv']),    'FedPDA'
};

gini_vals = zeros(1,5);

for idx = 1:5
    fname = sat_files{idx,1};
    sname = sat_files{idx,2};
    if ~isfile(fname), warning('%s missing, skip',fname); continue; end
    T = readtable(fname);
    vals = T.contributions;
    total_v = sum(vals);
    if total_v == 0, warning('%s sum=0, skip',sname); continue; end

    sv = sort(vals);
    n = numel(sv);
    cum = cumsum(sv) / total_v;
    xv = (1:n)' / n;

    plot(xv, cum, '-', 'Color',C.(sname), 'LineWidth',2, 'DisplayName',sname);

    ii = (1:n)';
    gini_vals(idx) = (2*sum(ii.*sv) - (n+1)*sum(sv)) / (n*sum(sv));
end

yp = 0.35;
for idx = 1:5
    sname = strats{idx};
    text(0.05, yp, sprintf('%s: Gini = %.3f', sname, gini_vals(idx)), ...
        'Units','normalized','FontSize',11,'FontWeight','bold','Color',C.(sname));
    yp = yp - 0.055;
end

xlabel('Cumulative Share of Satellites');
ylabel('Cumulative Share of Contributions');
title('Satellite Participation Fairness (\alpha=0.1)');
legend('Location','northwest'); xlim([0 1]); ylim([0 1]); axis square;
exportgraphics(gcf, fullfile(FIG_DIR,'fig6_fairness_lorenz_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 7 — FedSpace Dynamic Threshold (α=0.1)
%% ========================================================================
fprintf('[Fig 7] FedSpace Threshold (alpha=0.1)\n');
figure('Position',[100 100 1200 500]);

f = fullfile(DATA_DIR, [P 'fedspace_dynamic_threshold.csv']);
if isfile(f)
    T = readtable(f);

    subplot(1,2,1); hold on; grid on; box on;
    scatter(T.upcoming_contacts, T.threshold, 30, C.FedSpace, 'filled', ...
        'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','w','LineWidth',0.3);

    p = polyfit(T.upcoming_contacts, T.threshold, 1);
    xl = linspace(min(T.upcoming_contacts), max(T.upcoming_contacts), 100);
    plot(xl, polyval(p,xl), 'r--', 'LineWidth',1.5);

    R = corrcoef(T.upcoming_contacts, T.threshold);
    text(0.05,0.95, sprintf('\\itr = %.3f', R(1,2)), ...
        'Units','normalized','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    xlabel('Predicted Upcoming GS Contacts');
    ylabel('Dynamic Threshold');
    title('(a) Contact Prediction \rightarrow Threshold');

    subplot(1,2,2); hold on; grid on; box on;
    uniq_k = unique(T.buffer_size);
    k_cnt = arrayfun(@(v) sum(T.buffer_size==v), uniq_k);
    k_pct = k_cnt / height(T) * 100;

    bar(uniq_k, k_pct, 'FaceColor',C.FedSpace, 'FaceAlpha',0.85, 'EdgeColor','w');
    for j = 1:numel(k_pct)
        if k_pct(j) > 2
            text(uniq_k(j), k_pct(j)+1.5, sprintf('%.1f%%',k_pct(j)), ...
                'HorizontalAlignment','center','FontSize',9);
        end
    end

    xlabel('Buffer Size at Flush (K)');
    ylabel('Frequency (%)');
    title('(b) Effective Buffer Size Distribution');

    sgtitle('FedSpace Dynamic Threshold (\alpha=0.1)','FontSize',14);
end

exportgraphics(gcf, fullfile(FIG_DIR,'fig7_fedspace_threshold_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 8 — Summary Table (5전략, α=0.1)
%% ========================================================================
fprintf('[Fig 8] Summary Table (alpha=0.1)\n');
figure('Position',[100 100 1350 480]); axis off;

headers = {'Metric','FedAsync','FedBuff','FedSpace','FedOrbit','FedPDA'};
rows = {
    'Best Accuracy (%)',    '74.70',   '66.84',   '74.67',   '69.16',   '77.16'
    'Final Accuracy (%)',   '72.64',   '46.72',   '69.61',   '62.60',   '70.61'
    'Total Agg. Rounds',   '587',     '215',     '360',     '232',     '366'
    'Upload Rate (%)',      '10.0',    '36.5',    '36.2',    '5.2',     '36.2'
    'Gini Coefficient',     '0.126',   '0.027',   '0.024',   '0.029',   '0.024'
    'Participating Sats',   '225/238', '238/238', '238/238', '238/238', '238/238'
    'Avg Staleness',        '4.82',    '2.10',    '3.46',    '6.65',    '3.52'
    'Time to 70%',          '128.8h',  'N/A',     '96.1h',   'N/A',     '84.9h'
    'Late Std (%)',         '2.36',    '5.41',    '4.75',    '6.77',    '3.23'
};

nr = size(rows,1);
nc = 6;
cw = [0.18 0.15 0.15 0.15 0.15 0.15];
rh = 0.075;
x0 = 0.03; y0 = 0.86;

hdr_bg = cell(1,6);
hdr_bg{1} = [0.84 0.85 0.87];
hdr_bg{2} = [0.98 0.86 0.85];
hdr_bg{3} = [0.84 0.92 0.97];
hdr_bg{4} = [0.84 0.96 0.89];
hdr_bg{5} = [1.00 0.93 0.80];
hdr_bg{6} = [0.87 0.82 0.93];

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
        % Best accuracy → FedPDA (j=6)
        if i==1 && j==6, bg=[0.82 0.76 0.90]; fw='bold'; end
        % Best final accuracy → FedAsync (j=2)
        if i==2 && j==2, bg=[0.95 0.80 0.80]; fw='bold'; end
        % FedBuff final: 붕괴 경고 (j=3)
        if i==2 && j==3, bg=[1.0 0.85 0.80]; fw='bold'; end
        % Best Gini → FedSpace & FedPDA (j=4,6)
        if i==5 && (j==4 || j==6), bg=[0.85 0.93 0.85]; fw='bold'; end
        % Worst staleness → FedOrbit (j=5)
        if i==7 && j==5, bg=[1.0 0.85 0.80]; fw='bold'; end
        % Best 70% time → FedPDA (j=6)
        if i==8 && j==6, bg=[0.82 0.76 0.90]; fw='bold'; end
        % N/A → 회색
        if strcmp(str, 'N/A'), bg=[0.93 0.93 0.93]; end

        annotation('textbox',[x y cw(j) rh], ...
            'String',str,'FontWeight',fw,'FontSize',9.5, ...
            'HorizontalAlignment',ha,'VerticalAlignment','middle', ...
            'BackgroundColor',bg,'EdgeColor',[0.85 0.85 0.85],'Margin',2);
    end
end

title('Comparative Analysis Summary — \alpha=0.1 (Severe Non-IID)','FontSize',13,'FontWeight','bold');
exportgraphics(gcf, fullfile(FIG_DIR,'fig8_summary_table_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 9 — FedOrbit: Staleness & Plane (α=0.1)
%% ========================================================================
fprintf('[Fig 9] FedOrbit Detail (alpha=0.1)\n');
figure('Position',[100 100 1200 500]);

subplot(1,2,1); hold on; grid on; box on;
f = fullfile(DATA_DIR, [P 'fedorbit_staleness.csv']);
if isfile(f)
    T = readtable(f);
    fill([T.sim_hours; flipud(T.sim_hours)], ...
         [T.min; flipud(T.max)], ...
         C.FedOrbit, 'FaceAlpha',0.15, 'EdgeColor','none', ...
         'DisplayName','Min-Max Range');
    plot(T.sim_hours, T.mean, '-', 'Color',C.FedOrbit, 'LineWidth',0.8, ...
        'DisplayName','Mean Staleness');
    if height(T) > 20
        sm = movmean(T.mean, 50);
        plot(T.sim_hours, sm, '-', 'Color',[0.85 0.30 0.10], ...
            'LineWidth',2.5, 'DisplayName','Moving Avg (w=50)');
    end
    xlabel('Simulation Time (hours)'); ylabel('Staleness (\tau)');
    title('(a) Staleness Over Time'); legend('Location','northwest');
end

subplot(1,2,2); hold on; grid on; box on;
f = fullfile(DATA_DIR, [P 'fedorbit_plane_contributions_corrected.csv']);
if isfile(f)
    T = readtable(f);
    for k = 1:height(T)
        if T.contributions(k) == 0, bc = [0.90 0.30 0.20];
        else, bc = C.FedOrbit; end
        bar(T.plane_id(k), T.contributions(k), 0.7, ...
            'FaceColor',bc, 'FaceAlpha',0.85, 'EdgeColor','w');
    end
    active_vals = T.contributions(T.contributions > 0);
    if ~isempty(active_vals)
        yline(mean(active_vals), 'r--', 'LineWidth',1.5);
    end
    xlabel('Orbital Plane ID'); ylabel('Aggregation Contributions');
    title('(b) Per-Plane Contributions'); xticks(1:17);
end

sgtitle('FedOrbit: Staleness & Plane (\alpha=0.1)','FontSize',14);
exportgraphics(gcf, fullfile(FIG_DIR,'fig9_fedorbit_detail_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 10 — FedPDA: Staleness & Plane (α=0.1)
%% ========================================================================
fprintf('[Fig 10] FedPDA Detail (alpha=0.1)\n');
figure('Position',[100 100 1200 500]);

subplot(1,2,1); hold on; grid on; box on;
f = fullfile(DATA_DIR, [P 'fedpda_staleness.csv']);
if isfile(f)
    T = readtable(f);
    fill([T.sim_hours; flipud(T.sim_hours)], ...
         [T.min; flipud(T.max)], ...
         C.FedPDA, 'FaceAlpha',0.15, 'EdgeColor','none', ...
         'DisplayName','Min-Max Range');
    plot(T.sim_hours, T.mean, '-', 'Color',C.FedPDA, 'LineWidth',0.8, ...
        'DisplayName','Mean Staleness');
    if height(T) > 20
        sm = movmean(T.mean, 50);
        plot(T.sim_hours, sm, '-', 'Color',[0.85 0.30 0.10], ...
            'LineWidth',2.5, 'DisplayName','Moving Avg (w=50)');
    end
    xlabel('Simulation Time (hours)'); ylabel('Staleness (\tau)');
    title('(a) Staleness Over Time'); legend('Location','northwest');
end

subplot(1,2,2); hold on; grid on; box on;
f = fullfile(DATA_DIR, [P 'fedpda_plane_contributions_corrected.csv']);
if isfile(f)
    T = readtable(f);
    for k = 1:height(T)
        if T.contributions(k) == 0, bc = [0.90 0.30 0.20];
        else, bc = C.FedPDA; end
        bar(T.plane_id(k), T.contributions(k), 0.7, ...
            'FaceColor',bc, 'FaceAlpha',0.85, 'EdgeColor','w');
    end
    active_vals = T.contributions(T.contributions > 0);
    if ~isempty(active_vals)
        yline(mean(active_vals), 'r--', 'LineWidth',1.5);
    end
    xlabel('Orbital Plane ID'); ylabel('Aggregation Contributions');
    title('(b) Per-Plane Contributions'); xticks(1:17);
end

sgtitle('FedPDA: Staleness & Plane (\alpha=0.1)','FontSize',14);
exportgraphics(gcf, fullfile(FIG_DIR,'fig10_fedpda_detail_a01.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 11 — α=0.5 vs α=0.1 비교 테이블 (NEW)
%% ========================================================================
fprintf('[Fig 11] Alpha Comparison Table\n');
figure('Position',[100 100 1350 550]); axis off;

headers2 = {'', ...
    'FedAsync', 'FedAsync', ...
    'FedBuff', 'FedBuff', ...
    'FedSpace', 'FedSpace', ...
    'FedOrbit', 'FedOrbit', ...
    'FedPDA', 'FedPDA'};
subheaders = {'Metric', ...
    '\alpha=0.5', '\alpha=0.1', ...
    '\alpha=0.5', '\alpha=0.1', ...
    '\alpha=0.5', '\alpha=0.1', ...
    '\alpha=0.5', '\alpha=0.1', ...
    '\alpha=0.5', '\alpha=0.1'};

data2 = {
    'Best Acc. (%)', ...
        '80.42','74.70', '80.32','66.84', '82.60','74.67', '76.87','69.16', '82.01','77.16'
    'Final Acc. (%)', ...
        '79.55','72.64', '80.24','46.72', '80.09','69.61', '74.87','62.60', '81.62','70.61'
    '\Delta Best', ...
        '','-5.72', '','-13.48', '','-7.93', '','-7.71', '','-4.85'
    '\Delta Final', ...
        '','-6.91', '','-33.52', '','-10.48', '','-12.27', '','-11.01'
};

nr2 = size(data2,1);
nc2 = 11;
cw2 = [0.11, 0.08,0.08, 0.08,0.08, 0.08,0.08, 0.08,0.08, 0.08,0.08];
rh2 = 0.08;
x02 = 0.02; y02 = 0.82;

% 전략 헤더 (병합 효과)
strat_names = {'FedAsync','FedBuff','FedSpace','FedOrbit','FedPDA'};
strat_colors = {[0.98 0.86 0.85],[0.84 0.92 0.97],[0.84 0.96 0.89],[1.0 0.93 0.8],[0.87 0.82 0.93]};
for si = 1:5
    x_start = x02 + cw2(1) + sum(cw2(2:2*si-2+1));
    w_span = cw2(2*si) + cw2(2*si+1);
    annotation('textbox',[x_start y02+rh2 w_span rh2], ...
        'String',strat_names{si},'FontWeight','bold','FontSize',10, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'BackgroundColor',strat_colors{si},'EdgeColor',[0.7 0.7 0.7],'Margin',1);
end

% 서브 헤더 (α=0.5, α=0.1)
for j = 1:nc2
    x = x02 + sum(cw2(1:j-1));
    bg2 = [0.92 0.92 0.92];
    if j == 1, bg2 = [0.84 0.85 0.87]; end
    annotation('textbox',[x y02 cw2(j) rh2], ...
        'String',subheaders{j},'FontWeight','bold','FontSize',9, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'BackgroundColor',bg2,'EdgeColor',[0.7 0.7 0.7],'Margin',1);
end

% 데이터 행
for i = 1:nr2
    y = y02 - i*rh2;
    for j = 1:nc2
        x = x02 + sum(cw2(1:j-1));
        str = data2{i,j};
        if j == 1, fw = 'bold'; ha = 'left';
        else, fw = 'normal'; ha = 'center'; end

        bg2 = [1 1 1];
        % Δ 행에서 가장 큰 하락 강조
        if i==4 && j==5, bg2=[1.0 0.80 0.75]; fw='bold'; end  % FedBuff -33.52
        % 빈 칸 회색
        if isempty(str), bg2=[0.96 0.96 0.96]; str=' '; end

        annotation('textbox',[x y cw2(j) rh2], ...
            'String',str,'FontWeight',fw,'FontSize',9, ...
            'HorizontalAlignment',ha,'VerticalAlignment','middle', ...
            'BackgroundColor',bg2,'EdgeColor',[0.88 0.88 0.88],'Margin',1);
    end
end

title('\alpha=0.5 vs \alpha=0.1: Non-IID Sensitivity Analysis','FontSize',13,'FontWeight','bold');
exportgraphics(gcf, fullfile(FIG_DIR,'fig11_alpha_comparison.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
fprintf('\n==============================\n');
fprintf(' 11개 그래프 생성 완료 (alpha=0.1)\n');
fprintf(' 저장 위치: %s/\n', FIG_DIR);
fprintf('==============================\n');
