%% ========================================================================
%  FedPDA η_g Sweep — 개별 케이스 상세 분석
%  ========================================================================
%  6개 케이스 각각에 대해 3개 subplot (Acc+Staleness+Plane)
%  하나의 스크립트에서 모든 케이스 순회
%  ========================================================================

clear; clc; close all;

%% ========================= 경로 설정 ===================================
DATA_DIR = './data';
FIG_DIR  = './figures_cases';
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontSize', 10);

%% ========================= 케이스 정의 ==================================
cases = {
    0.1, 0.1, [0.906 0.298 0.235]   % α=0.1, η_g=0.1
    0.1, 0.3, [0.945 0.600 0.090]   % α=0.1, η_g=0.3
    0.1, 0.5, [0.580 0.404 0.741]   % α=0.1, η_g=0.5
    0.5, 0.1, [0.906 0.298 0.235]   % α=0.5, η_g=0.1
    0.5, 0.3, [0.945 0.600 0.090]   % α=0.5, η_g=0.3
    0.5, 0.5, [0.580 0.404 0.741]   % α=0.5, η_g=0.5
};

for ci = 1:size(cases,1)
    alpha = cases{ci,1};
    eta   = cases{ci,2};
    clr   = cases{ci,3};
    pres  = (1 - eta) * 100;

    prefix = sprintf('a%g_eta%g', alpha, eta);
    case_label = sprintf('\\alpha=%.1f, \\eta_g=%.1f (%d%% preservation)', alpha, eta, round(pres));

    fprintf('[Case %d] %s\n', ci, prefix);

    % 데이터 로드
    acc_file   = fullfile(DATA_DIR, [prefix '_accuracy.csv']);
    stale_file = fullfile(DATA_DIR, [prefix '_staleness.csv']);
    plane_file = fullfile(DATA_DIR, [prefix '_plane_contributions.csv']);

    if ~isfile(acc_file), warning('Missing: %s', acc_file); continue; end

    T_acc   = readtable(acc_file, 'VariableNamingRule','preserve');
    T_stale = readtable(stale_file, 'VariableNamingRule','preserve');
    T_plane = readtable(plane_file, 'VariableNamingRule','preserve');

    figure('Position',[50 50 1400 450]);

    %% --- (a) Accuracy & Loss vs Hours ---
    subplot(1,3,1); hold on; grid on; box on;

    yyaxis left;
    mk = max(1, floor(height(T_acc)/20));
    plot(T_acc.sim_hours, T_acc.accuracy, '-', ...
        'Color',clr, 'LineWidth',1.5, ...
        'Marker','o', 'MarkerSize',4, ...
        'MarkerIndices',1:mk:height(T_acc), ...
        'MarkerFaceColor',clr);
    ylabel('Accuracy (%)');

    % Best/Final 표시
    [best_acc, best_idx] = max(T_acc.accuracy);
    plot(T_acc.sim_hours(best_idx), best_acc, 'p', ...
        'MarkerSize',14, 'MarkerFaceColor','r', 'MarkerEdgeColor','k');
    text(T_acc.sim_hours(best_idx)+2, best_acc, ...
        sprintf('Best: %.1f%%', best_acc), 'FontSize',9, 'FontWeight','bold');

    yyaxis right;
    plot(T_acc.sim_hours, T_acc.loss, '--', ...
        'Color',[0.5 0.5 0.5], 'LineWidth',1);
    ylabel('Loss');

    xlabel('Simulation Time (hours)');
    title('(a) Accuracy & Loss');
    ax = gca;
    ax.YAxis(1).Color = clr;
    ax.YAxis(2).Color = [0.5 0.5 0.5];

    %% --- (b) Staleness Over Time ---
    subplot(1,3,2); hold on; grid on; box on;

    fill([T_stale.sim_hours; flipud(T_stale.sim_hours)], ...
         [T_stale.min; flipud(T_stale.max)], ...
         clr, 'FaceAlpha',0.15, 'EdgeColor','none', ...
         'DisplayName','Min-Max Range');
    plot(T_stale.sim_hours, T_stale.mean, '-', ...
        'Color',clr, 'LineWidth',0.8, 'DisplayName','Mean');

    if height(T_stale) > 20
        sm = movmean(T_stale.mean, 50);
        plot(T_stale.sim_hours, sm, '-', 'Color',[0.85 0.30 0.10], ...
            'LineWidth',2.5, 'DisplayName','Moving Avg (w=50)');
    end

    xlabel('Simulation Time (hours)'); ylabel('Staleness (\tau)');
    title('(b) Staleness');
    legend('Location','northwest','FontSize',8);

    %% --- (c) Per-Plane Contributions ---
    subplot(1,3,3); hold on; grid on; box on;

    for k = 1:height(T_plane)
        if T_plane.contributions(k) == 0
            bc = [0.90 0.30 0.20];
        else
            bc = clr;
        end
        bar(T_plane.plane_id(k), T_plane.contributions(k), 0.7, ...
            'FaceColor',bc, 'FaceAlpha',0.85, 'EdgeColor','w');
    end

    active_vals = T_plane.contributions(T_plane.contributions > 0);
    if ~isempty(active_vals)
        yline(mean(active_vals), 'r--', 'LineWidth',1.5, 'DisplayName','Mean');
        legend('Location','northeast','FontSize',8);
    end

    xlabel('Orbital Plane ID'); ylabel('Contributions');
    title('(c) Per-Plane');
    xticks(1:17);

    %% --- 전체 제목 & 저장 ---
    sgtitle(sprintf('FedPDA Detail: %s', case_label), 'FontSize',13);

    out_name = sprintf('case_%s.png', prefix);
    exportgraphics(gcf, fullfile(FIG_DIR, out_name), 'Resolution',300);
    fprintf('  → %s\n', out_name);
end

%% ========================================================================
fprintf('\n==============================\n');
fprintf(' 6개 케이스 상세 분석 완료\n');
fprintf(' 저장 위치: %s/\n', FIG_DIR);
fprintf('==============================\n');
