%% ========================================================================
%  FedPDA η_g Sweep — α=0.1 vs α=0.5 교차 비교
%  ========================================================================
%  Fig 1: η_g별 Best/Final Accuracy (α 비교, Grouped Bar)
%  Fig 2: η_g별 Late-stage Std (안정성 비교)
%  Fig 3: η_g별 Time-to-70% (수렴 속도)
%  Fig 4: α=0.1 vs α=0.5 비교 테이블
%  Fig 5: U자형 트레이드오프 (Best Acc vs η_g, 양 α)
%  ========================================================================

clear; clc; close all;

%% ========================= 경로 설정 ===================================
DATA_DIR = './data';
FIG_DIR  = './figures_cross';
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

%% ========================= 색상 ========================================
C_a01 = [0.906, 0.298, 0.235];   % 빨강 (α=0.1)
C_a05 = [0.204, 0.596, 0.859];   % 파랑 (α=0.5)

set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);

%% ========================= 데이터 ======================================
% 요약 테이블 로드
S = readtable(fullfile(DATA_DIR, 'summary.csv'), 'VariableNamingRule','preserve');

% α=0.1, α=0.5 분리
S01 = S(S.alpha == 0.1, :);
S05 = S(S.alpha == 0.5, :);

eta_vals = S01.eta_g;
x_pos = 1:numel(eta_vals);
x_labels = arrayfun(@(v) sprintf('\\eta_g=%.1f (%d%%)', v, round((1-v)*100)), ...
    eta_vals, 'UniformOutput', false);

%% ========================================================================
%  Figure 1 — Best & Final Accuracy (Grouped Bar)
%% ========================================================================
fprintf('[Fig 1] Best & Final Accuracy Comparison\n');
figure('Position',[100 100 1000 500]);

subplot(1,2,1); hold on; grid on; box on;
bar_data = [S01.best_acc, S05.best_acc];
b = bar(x_pos, bar_data, 'grouped');
b(1).FaceColor = C_a01; b(1).FaceAlpha = 0.85;
b(2).FaceColor = C_a05; b(2).FaceAlpha = 0.85;

for i = 1:numel(x_pos)
    text(x_pos(i)-0.15, bar_data(i,1)+0.8, sprintf('%.1f',bar_data(i,1)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    text(x_pos(i)+0.15, bar_data(i,2)+0.8, sprintf('%.1f',bar_data(i,2)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

set(gca,'XTick',x_pos,'XTickLabel',x_labels);
ylabel('Best Accuracy (%)');
title('(a) Best Accuracy');
legend('\alpha=0.1','\alpha=0.5','Location','southeast');
ylim([60 90]);

subplot(1,2,2); hold on; grid on; box on;
bar_data2 = [S01.final_acc, S05.final_acc];
b2 = bar(x_pos, bar_data2, 'grouped');
b2(1).FaceColor = C_a01; b2(1).FaceAlpha = 0.85;
b2(2).FaceColor = C_a05; b2(2).FaceAlpha = 0.85;

for i = 1:numel(x_pos)
    text(x_pos(i)-0.15, bar_data2(i,1)+0.8, sprintf('%.1f',bar_data2(i,1)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    text(x_pos(i)+0.15, bar_data2(i,2)+0.8, sprintf('%.1f',bar_data2(i,2)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

set(gca,'XTick',x_pos,'XTickLabel',x_labels);
ylabel('Final Accuracy (%)');
title('(b) Final Accuracy');
legend('\alpha=0.1','\alpha=0.5','Location','southeast');
ylim([60 90]);

sgtitle('FedPDA: Best & Final Accuracy — \alpha=0.1 vs \alpha=0.5','FontSize',13);
exportgraphics(gcf, fullfile(FIG_DIR,'fig1_best_final_acc.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 2 — Late-stage Std (안정성)
%% ========================================================================
fprintf('[Fig 2] Late-stage Stability\n');
figure('Position',[100 100 700 500]); hold on; grid on; box on;

bar_data3 = [S01.late_std, S05.late_std];
b3 = bar(x_pos, bar_data3, 'grouped');
b3(1).FaceColor = C_a01; b3(1).FaceAlpha = 0.85;
b3(2).FaceColor = C_a05; b3(2).FaceAlpha = 0.85;

for i = 1:numel(x_pos)
    text(x_pos(i)-0.15, bar_data3(i,1)+0.05, sprintf('%.2f',bar_data3(i,1)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    text(x_pos(i)+0.15, bar_data3(i,2)+0.05, sprintf('%.2f',bar_data3(i,2)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

set(gca,'XTick',x_pos,'XTickLabel',x_labels);
ylabel('Late-stage Accuracy Std (%)');
title('FedPDA: Late-stage Stability — \alpha=0.1 vs \alpha=0.5');
legend('\alpha=0.1','\alpha=0.5','Location','northeast');
exportgraphics(gcf, fullfile(FIG_DIR,'fig2_late_std.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 3 — Time to 70%
%% ========================================================================
fprintf('[Fig 3] Time to 70%%\n');
figure('Position',[100 100 700 500]); hold on; grid on; box on;

t70_01 = S01.time_to_70pct;
t70_05 = S05.time_to_70pct;

% N/A 처리: string → double 변환
if iscell(t70_01) || isstring(t70_01)
    t70_01 = cellfun(@(x) str2double(x), cellstr(t70_01));
end
if iscell(t70_05) || isstring(t70_05)
    t70_05 = cellfun(@(x) str2double(x), cellstr(t70_05));
end

bar_data4 = [t70_01, t70_05];
b4 = bar(x_pos, bar_data4, 'grouped');
b4(1).FaceColor = C_a01; b4(1).FaceAlpha = 0.85;
b4(2).FaceColor = C_a05; b4(2).FaceAlpha = 0.85;

for i = 1:numel(x_pos)
    for j = 1:2
        v = bar_data4(i,j);
        if ~isnan(v)
            xoff = (j-1.5)*0.3;
            text(x_pos(i)+xoff, v+3, sprintf('%.1fh',v), ...
                'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
        end
    end
end

set(gca,'XTick',x_pos,'XTickLabel',x_labels);
ylabel('Time to 70% Accuracy (hours)');
title('FedPDA: Convergence Speed — \alpha=0.1 vs \alpha=0.5');
legend('\alpha=0.1','\alpha=0.5','Location','northeast');
exportgraphics(gcf, fullfile(FIG_DIR,'fig3_time_to_70.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 4 — 비교 테이블
%% ========================================================================
fprintf('[Fig 4] Cross Comparison Table\n');
figure('Position',[100 100 1200 480]); axis off;

headers = {'', ...
    '\eta_g=0.1', '\eta_g=0.1', ...
    '\eta_g=0.3', '\eta_g=0.3', ...
    '\eta_g=0.5', '\eta_g=0.5'};
subheaders = {'Metric', ...
    '\alpha=0.1', '\alpha=0.5', ...
    '\alpha=0.1', '\alpha=0.5', ...
    '\alpha=0.1', '\alpha=0.5'};

data = {
    'Best Acc. (%)', ...
        '70.12','76.13', '76.47','80.97', '77.44','82.01'
    'Final Acc. (%)', ...
        '67.68','76.01', '70.99','80.82', '70.97','81.71'
    'Late Mean (%)', ...
        '68.00','74.34', '73.80','79.80', '73.91','80.85'
    'Late Std (%)', ...
        '1.33','1.15', '1.52','0.82', '2.17','0.73'
    'Time to 70% (h)', ...
        '165.7','94.3', '94.3','52.6', '84.9','41.2'
    'Time to 80% (h)', ...
        'N/A','N/A', 'N/A','138.1', 'N/A','108.1'
};

nr = size(data,1);
nc = 7;
cw = [0.18, 0.12,0.12, 0.12,0.12, 0.12,0.12];
rh = 0.09;
x0 = 0.03; y0 = 0.78;

% η_g 그룹 헤더
eta_names = {'\eta_g=0.1 (90%)','\eta_g=0.3 (70%)','\eta_g=0.5 (50%)'};
eta_colors = {[0.98 0.86 0.85],[1.0 0.93 0.8],[0.87 0.82 0.93]};
for si = 1:3
    x_start = x0 + cw(1) + sum(cw(2:2*si-1));
    w_span = cw(2*si) + cw(2*si+1);
    annotation('textbox',[x_start y0+rh w_span rh], ...
        'String',eta_names{si},'FontWeight','bold','FontSize',10, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'BackgroundColor',eta_colors{si},'EdgeColor',[0.7 0.7 0.7],'Margin',1);
end

% 서브헤더
for j = 1:nc
    x = x0 + sum(cw(1:j-1));
    bg = [0.92 0.92 0.92];
    if j == 1, bg = [0.84 0.85 0.87]; end
    annotation('textbox',[x y0 cw(j) rh], ...
        'String',subheaders{j},'FontWeight','bold','FontSize',9, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'BackgroundColor',bg,'EdgeColor',[0.7 0.7 0.7],'Margin',1);
end

% 데이터
for i = 1:nr
    y = y0 - i*rh;
    for j = 1:nc
        x = x0 + sum(cw(1:j-1));
        str = data{i,j};
        if j == 1, fw = 'bold'; ha = 'left';
        else, fw = 'normal'; ha = 'center'; end

        bg = [1 1 1];
        % Best overall → α=0.5, η_g=0.5
        if i==1 && j==7, bg=[0.82 0.76 0.90]; fw='bold'; end
        if i==2 && j==7, bg=[0.82 0.76 0.90]; fw='bold'; end
        % N/A
        if strcmp(str,'N/A'), bg=[0.93 0.93 0.93]; end

        annotation('textbox',[x y cw(j) rh], ...
            'String',str,'FontWeight',fw,'FontSize',9, ...
            'HorizontalAlignment',ha,'VerticalAlignment','middle', ...
            'BackgroundColor',bg,'EdgeColor',[0.88 0.88 0.88],'Margin',1);
    end
end

title('FedPDA \eta_g Sweep: \alpha=0.1 vs \alpha=0.5','FontSize',13,'FontWeight','bold');
exportgraphics(gcf, fullfile(FIG_DIR,'fig4_cross_table.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
%  Figure 5 — U자형 트레이드오프 (Best Acc vs η_g)
%% ========================================================================
fprintf('[Fig 5] U-curve Tradeoff\n');
figure('Position',[100 100 800 500]); hold on; grid on; box on;

% 데이터: session_summary.md의 η_g=0.7, 1.0 포함
% α=0.1: η_g = [0.1, 0.3, 0.5, 0.7, 1.0]
eta_full_01 = [0.1, 0.3, 0.5, 0.7, 1.0];
best_full_01 = [70.12, 76.47, 77.44, 77.16, 75.31];
final_full_01 = [67.68, 70.99, 70.97, 70.61, 67.82];

% α=0.5: η_g = [0.1, 0.3, 0.5, 0.7]  (1.0 = 기존 baseline)
eta_full_05 = [0.1, 0.3, 0.5, 0.7];
best_full_05 = [76.13, 80.97, 82.01, 82.01];  % 0.7은 session에서 FedPDA+ISL α=0.5 기준

plot(eta_full_01, best_full_01, '-o', ...
    'Color',C_a01, 'LineWidth',2, 'MarkerSize',8, ...
    'MarkerFaceColor',C_a01, 'DisplayName','\alpha=0.1 (Best Acc)');
plot(eta_full_05, best_full_05, '-s', ...
    'Color',C_a05, 'LineWidth',2, 'MarkerSize',8, ...
    'MarkerFaceColor',C_a05, 'DisplayName','\alpha=0.5 (Best Acc)');

% 최적점 강조
[~, idx01] = max(best_full_01);
plot(eta_full_01(idx01), best_full_01(idx01), 'p', ...
    'MarkerSize',18, 'MarkerFaceColor','r', 'MarkerEdgeColor','k', ...
    'DisplayName',sprintf('\\alpha=0.1 optimal (\\eta_g=%.1f)', eta_full_01(idx01)));

[~, idx05] = max(best_full_05);
plot(eta_full_05(idx05), best_full_05(idx05), 'p', ...
    'MarkerSize',18, 'MarkerFaceColor',[0.2 0.7 0.3], 'MarkerEdgeColor','k', ...
    'DisplayName',sprintf('\\alpha=0.5 optimal (\\eta_g=%.1f)', eta_full_05(idx05)));

xlabel('\eta_g (Global Preservation Rate = 1 - \eta_g)');
ylabel('Best Accuracy (%)');
title('FedPDA: \eta_g Trade-off (U-shaped)');
legend('Location','southwest');
xlim([0 1.1]); ylim([65 85]);

% 보존률 축 (상단)
ax1 = gca;
ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', ...
    'YAxisLocation','right', 'Color','none');
ax2.XLim = [0 1.1];
ax2.XTick = [0.1 0.3 0.5 0.7 1.0];
ax2.XTickLabel = {'90%','70%','50%','30%','0%'};
ax2.XLabel.String = 'Global Preservation Rate';
ax2.YTick = [];
linkaxes([ax1, ax2], 'x');

exportgraphics(gcf, fullfile(FIG_DIR,'fig5_u_curve.png'), 'Resolution',300);
fprintf('  OK\n');

%% ========================================================================
fprintf('\n==============================\n');
fprintf(' 5개 그래프 생성 완료 (교차 비교)\n');
fprintf(' 저장 위치: %s/\n', FIG_DIR);
fprintf('==============================\n');
