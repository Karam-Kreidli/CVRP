import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:RouteIQ_UI/screens/solution_screen.dart';
import 'package:RouteIQ_UI/services/benchmark_service.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/widgets/Tag.dart';
import 'package:RouteIQ_UI/widgets/sectionlabel_line.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ─── ROUTE OBSERVER ──────────────────────────────────────────────────────────

/// Register this in [MaterialApp.navigatorObservers] so that [BenchmarkScreen]
/// can reset itself whenever the user navigates back to it.
final RouteObserver<ModalRoute<void>> newBenchmarkRouteObserver =
    RouteObserver<ModalRoute<void>>();

// ─── BENCHMARK SCREEN ────────────────────────────────────────────────────────

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> with RouteAware {
  final _svc = BenchmarkService.instance;

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final route = ModalRoute.of(context);
    if (route != null) newBenchmarkRouteObserver.subscribe(this, route);
  }

  @override
  void dispose() {
    newBenchmarkRouteObserver.unsubscribe(this);
    super.dispose();
  }

  // ── Actions ───────────────────────────────────────────────────────────────

  void _cancelBenchmark() {
    // Currently no way to cancel — could add later if needed
  }

  void _reBenchmark() {
    _svc.retryFetch();
  }

  // ─── BUILD ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg0,
      child: Stack(
        children: [
          const _GridBackground(),
          ListenableBuilder(
            listenable: _svc,
            builder: (context, _) {
              return SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(32, 28, 32, 40),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildHeader(),
                    const SizedBox(height: 28),
                    _buildBody(context),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // ─── HEADER ──────────────────────────────────────────────────────────────

  Widget _buildHeader() {
    final compact = MediaQuery.of(context).size.width < 1125;
    final result = _svc.result;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (!compact) ...[
          Row(
            children: [
              Flexible(
                child: Text(
                  'Benchmark Comparison',
                  style: AppTextStyles.displayMedium,
                ),
              ),
              const SizedBox(width: 12),
              if (result != null) ...[
                Tag(
                  label: result.rlWins ? 'RL WINS' : 'MIXED RESULTS',
                  color: result.rlWins ? AppColors.green : AppColors.amber,
                  fixWidth: false,
                ),
                const SizedBox(width: 8),
                Tag(
                  label: result.instanceName.toUpperCase(),
                  color: AppColors.purple,
                  fixWidth: false,
                ),
              ],
            ],
          ),
        ] else ...[
          Text('Benchmark Comparison', style: AppTextStyles.displayMedium),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              if (result != null) ...[
                Tag(
                  label: result.rlWins ? 'RL WINS' : 'MIXED RESULTS',
                  color: result.rlWins ? AppColors.green : AppColors.amber,
                  fixWidth: false,
                ),
                Tag(
                  label: result.instanceName.toUpperCase(),
                  color: AppColors.purple,
                  fixWidth: false,
                ),
              ],
            ],
          ),
        ],
        const SizedBox(height: 6),
        Text(
          'RouteIQ RL Agent vs HGS Default vs HGS Large Population'
          ' · uploaded instance · same iteration budget',
          style: AppTextStyles.mono,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  // ─── BODY ────────────────────────────────────────────────────────────────

  Widget _buildBody(BuildContext context) {
    if (!_svc.hasFile) return _buildEmptyState();
    if (_svc.isRunning) return _buildLoadingState();
    if (_svc.errorMessage != null) return _buildErrorState(_svc.errorMessage!);
    if (_svc.result != null) return _buildResults(context, _svc.result!);
    return _buildNoJobState();
  }

  // ─── EMPTY STATE ──────────────────────────────────────────────────────────

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 80),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _StateIcon(
              icon: Icons.upload_file_outlined,
              color: AppColors.textMuted,
            ),
            const SizedBox(height: 20),
            Text(
              'No Instance Loaded',
              style: AppTextStyles.subheading.copyWith(
                color: AppColors.textPrimary,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload a .vrp file in the Solver Console and run a solve first.',
              style: AppTextStyles.mono.copyWith(
                color: AppColors.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            _OutlineButton(
              label: 'GO TO SOLVER CONSOLE',
              icon: Icons.play_arrow,
              color: AppColors.cyan,
              onTap: () => ShellNav.push(RouteName.runSolver),
            ),
          ],
        ),
      ),
    );
  }

  // ─── NO JOB STATE ─────────────────────────────────────────────────────────

  Widget _buildNoJobState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 80),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _StateIcon(icon: Icons.show_chart_outlined, color: AppColors.cyan),
            const SizedBox(height: 20),
            Text(
              'Ready to Benchmark',
              style: AppTextStyles.subheading.copyWith(
                color: AppColors.textPrimary,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Run a solve in the Solver Console first to generate a job ID,\n'
              'then the benchmark comparison will load automatically.',
              style: AppTextStyles.mono.copyWith(
                color: AppColors.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            _OutlineButton(
              label: 'GO TO SOLVER CONSOLE',
              icon: Icons.play_arrow,
              color: AppColors.cyan,
              onTap: () => ShellNav.push(RouteName.runSolver),
            ),
          ],
        ),
      ),
    );
  }

  // ─── LOADING STATE ────────────────────────────────────────────────────────

  Widget _buildLoadingState() {
    return _BenchCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SectionLabel(title: 'BENCHMARK LOADING'),
          const SizedBox(height: 20),
          const SizedBox(
            width: 32,
            height: 32,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: AlwaysStoppedAnimation<Color>(AppColors.cyan),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Fetching benchmark results for your solve job…',
            style: AppTextStyles.mono.copyWith(color: AppColors.textSecondary),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            'If the benchmark is still computing, this will retry automatically.',
            style: AppTextStyles.monoSmall,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: const LinearProgressIndicator(
              minHeight: 3,
              backgroundColor: AppColors.border,
              valueColor: AlwaysStoppedAnimation<Color>(AppColors.cyan),
            ),
          ),
          const SizedBox(height: 20),
          _OutlineButton(
            label: 'TERMINATE',
            icon: Icons.stop,
            color: AppColors.red,
            onTap: _cancelBenchmark,
          ),
        ],
      ),
    );
  }

  // ─── ERROR STATE ──────────────────────────────────────────────────────────

  Widget _buildErrorState(String message) {
    return _BenchCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SectionLabel(title: 'BENCHMARK ERROR'),
          const SizedBox(height: 16),
          const Icon(Icons.error_outline, color: AppColors.red, size: 32),
          const SizedBox(height: 12),
          Text(
            message,
            style: AppTextStyles.mono.copyWith(color: AppColors.red),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          _OutlineButton(
            label: 'RETRY',
            icon: Icons.refresh,
            color: AppColors.amber,
            onTap: _reBenchmark,
          ),
        ],
      ),
    );
  }

  // ─── RESULTS ──────────────────────────────────────────────────────────────

  Widget _buildResults(BuildContext context, BenchmarkResult result) {
    final compact = MediaQuery.of(context).size.width < 1100;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildKpiRow(context, result),
        const SizedBox(height: 16),
        if (compact) ...[
          _buildBarChart(result),
          const SizedBox(height: 16),
          _buildResultsTable(result),
        ] else
          IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(flex: 5, child: _buildBarChart(result)),
                const SizedBox(width: 16),
                Expanded(flex: 7, child: _buildResultsTable(result)),
              ],
            ),
          ),
        const SizedBox(height: 20),
        _OutlineButton(
          label: 'RE-BENCHMARK',
          icon: Icons.refresh,
          color: AppColors.textSecondary,
          onTap: _svc.completedJobId != null ? _reBenchmark : null,
        ),
      ],
    );
  }

  // ── KPI Score Cards ───────────────────────────────────────────────────────

  Widget _buildKpiRow(BuildContext context, BenchmarkResult r) {
    final compact = MediaQuery.of(context).size.width < 1125;

    final rlCard = _BenchKpiCard(
      label: 'RL Avg Score',
      value: _fmtInt(r.rl.score.round()),
      sub: 'RouteIQ RL agent (greedy)',
      color: AppColors.cyan,
      borderColor: AppColors.cyan,
    );
    final defaultHgsCard = _BenchKpiCard(
      label: 'Default HGS Score',
      value: r.pctVsDefault > 0
          ? '↓ ${r.pctVsDefault.toStringAsFixed(1)}%'
          : '—',
      sub: 'avg score reduction vs HGS Default',
      color: AppColors.green,
      tag:
          '+${r.nvSavedVsDefault} vehicle${r.nvSavedVsDefault == 1 ? '' : 's'} eliminated',
      tagColor: AppColors.green,
    );
    final largePopCard = _BenchKpiCard(
      label: 'Large Population HGS Score',
      value: r.pctVsLargePop > 0
          ? '↓ ${r.pctVsLargePop.toStringAsFixed(1)}%'
          : '—',
      sub: 'mu=50, lambda=80 baseline',
      color: AppColors.green,
    );
    final nvImpactCard = _BenchKpiCard(
      label: 'NV Impact on Score',
      value: r.nvSavedVsDefault > 0 ? _fmtInt(r.nvSavedVsDefault * 1000) : '—',
      sub: 'pts saved from ${r.nvSavedVsDefault} fewer vehicles',
      color: AppColors.textPrimary,
      tag: '${r.nvSavedVsDefault} trucks × 1,000',
      tagColor: AppColors.textSecondary,
    );

    if (compact) {
      return Column(
        children: [
          SizedBox(width: double.infinity, child: rlCard),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: defaultHgsCard),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: largePopCard),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: nvImpactCard),
        ],
      );
    }

    return Row(
      children: [
        Expanded(child: rlCard),
        const SizedBox(width: 12),
        Expanded(child: defaultHgsCard),
        const SizedBox(width: 12),
        Expanded(child: largePopCard),
        const SizedBox(width: 12),
        Expanded(child: nvImpactCard),
      ],
    );
  }

  // ── Bar Chart ─────────────────────────────────────────────────────────────

  Widget _buildBarChart(BenchmarkResult result) {
    final entries = <_BarEntry>[
      _BarEntry(
        name: 'RouteIQ RL',
        score: result.rl.score,
        color: AppColors.cyan,
        improvementPct: null,
      ),
      _BarEntry(
        name: 'HGS Default',
        score: result.hgsDefault.score,
        color: AppColors.amber,
        improvementPct: result.pctVsDefault > 0 ? result.pctVsDefault : null,
      ),
      _BarEntry(
        name: 'HGS Large Pop',
        score: result.hgsLargePop.score,
        color: AppColors.red,
        improvementPct: result.pctVsLargePop > 0 ? result.pctVsLargePop : null,
      ),
    ];

    return _BenchCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SectionLabel(title: 'SCORE COMPARISON — PER SOLVER'),
          const SizedBox(height: 4),
          Text(
            'LOWER = BETTER  ·  hover a bar to inspect',
            style: AppTextStyles.monoSmall.copyWith(
              color: AppColors.textMuted,
              fontSize: 9,
            ),
          ),
          const SizedBox(height: 20),
          SizedBox(height: 220, child: _ScoreBarChart(entries: entries)),
        ],
      ),
    );
  }

  // ── Results Table ─────────────────────────────────────────────────────────

  Widget _buildResultsTable(BenchmarkResult result) {
    final compact = MediaQuery.of(context).size.width < 1100;

    final rows = [
      (name: 'RouteIQ RL', algo: result.rl, color: AppColors.cyan),
      (name: 'HGS Default', algo: result.hgsDefault, color: AppColors.amber),
      (name: 'HGS Large Pop', algo: result.hgsLargePop, color: AppColors.red),
    ];

    final bestScore = [
      result.rl.score,
      result.hgsDefault.score,
      result.hgsLargePop.score,
    ].reduce(math.min);

    final cols = compact
        ? const ['SOLVER', 'SCORE', 'VEHICLES (NV)']
        : const [
            'SOLVER',
            'SCORE',
            'VEHICLES (NV)',
            'DISTANCE (TD)',
            'SOLVE TIME',
          ];

    return _BenchCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SectionLabel(title: 'PER-SOLVER RESULTS — ALL METRICS'),
          const SizedBox(height: 14),
          _TableRow(cells: cols, isHeader: true),
          Container(
            height: 1,
            color: AppColors.border,
            margin: const EdgeInsets.symmetric(vertical: 6),
          ),
          for (final row in rows)
            _TableRow(
              cells: compact
                  ? [row.name, _fmtScore(row.algo.score), '${row.algo.nv}']
                  : [
                      row.name,
                      _fmtScore(row.algo.score),
                      '${row.algo.nv}',
                      row.algo.td.toStringAsFixed(1),
                      row.algo.solveTimeSeconds != null
                          ? '${row.algo.solveTimeSeconds}s'
                          : '—',
                    ],
              accentColor: row.color,
              isBest: row.algo.score == bestScore,
            ),
        ],
      ),
    );
  }

  // ─── FORMATTING ────────────────────────────────────────────────────────────

  String _fmtInt(int v) {
    final s = v.toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  String _fmtScore(double v) => _fmtInt(v.round());
}

// ─── BAR ENTRY MODEL ──────────────────────────────────────────────────────────

class _BarEntry {
  final String name;
  final double score;
  final Color color;
  final double? improvementPct;

  const _BarEntry({
    required this.name,
    required this.score,
    required this.color,
    required this.improvementPct,
  });
}

// ─── SCORE BAR CHART ─────────────────────────────────────────────────────────

class _ScoreBarChart extends StatefulWidget {
  final List<_BarEntry> entries;
  const _ScoreBarChart({required this.entries});

  @override
  State<_ScoreBarChart> createState() => _ScoreBarChartState();
}

class _ScoreBarChartState extends State<_ScoreBarChart>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _growAnim;
  late final Animation<double> _fadeAnim;
  int? _hoveredIndex;

  static const _padL = 54.0;
  static const _padR = 16.0;
  static const _padT = 32.0; // room for improvement badges
  static const _padB = 44.0;
  static const _barFrac = 0.55; // bar width as fraction of slot width

  double _computeTickStep(List<double> values) {
    const eps = 1e-12;
    final sorted = [...values]..sort();
    double? minGap;

    for (int i = 1; i < sorted.length; i++) {
      final gap = (sorted[i] - sorted[i - 1]).abs();
      if (gap <= eps) continue;
      if (minGap == null || gap < minGap) minGap = gap;
    }

    if (minGap == null) {
      final base = sorted.isEmpty ? 1.0 : math.max(sorted.first.abs(), 1.0);
      return base * 0.01;
    }

    return double.parse(minGap.toStringAsPrecision(10));
  }

  int _decimalsForStep(double step) {
    const eps = 1e-9;
    int decimals = 0;
    double scaled = step.abs();
    while (decimals < 8 && (scaled - scaled.roundToDouble()).abs() > eps) {
      scaled *= 10;
      decimals++;
    }
    return decimals;
  }

  String _formatLikeTable(double value) {
    final s = value.round().toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _growAnim = CurvedAnimation(
      parent: _ctrl,
      curve: const Interval(0.0, 0.80, curve: Curves.easeOutCubic),
    );
    _fadeAnim = CurvedAnimation(
      parent: _ctrl,
      curve: const Interval(0.50, 1.0, curve: Curves.easeOut),
    );
    _ctrl.forward();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final entries = widget.entries;
    if (entries.isEmpty) return const SizedBox.shrink();

    final scores = entries.map((e) => e.score).toList();
    final rawMax = scores.reduce(math.max);
    final rawMin = scores.reduce(math.min);
    final yTickStep = _computeTickStep(scores);
    final yLabelDecimals = _decimalsForStep(yTickStep);

    final chartMin = (rawMin / yTickStep).floor() * yTickStep;
    var chartMax = (rawMax / yTickStep).ceil() * yTickStep;
    if ((chartMax - rawMax).abs() < 1e-10) {
      chartMax += yTickStep;
    }
    if (chartMax <= chartMin) {
      chartMax = chartMin + yTickStep;
    }

    return AnimatedBuilder(
      animation: _ctrl,
      builder: (context, _) {
        return LayoutBuilder(
          builder: (context, constraints) {
            final totalW = constraints.maxWidth;
            final totalH = constraints.maxHeight;
            final chartW = totalW - _padL - _padR;
            final chartH = totalH - _padT - _padB;
            final n = entries.length;
            final slotW = chartW / n;
            final barW = slotW * _barFrac;

            // Compute bar rects
            final barRects = <Rect>[];
            for (int i = 0; i < n; i++) {
              final frac = chartMax > chartMin
                  ? (entries[i].score - chartMin) / (chartMax - chartMin)
                  : 0.5;
              final barH = frac.clamp(0.0, 1.0) * chartH * _growAnim.value;
              final cx = _padL + slotW * i + slotW / 2;
              barRects.add(
                Rect.fromLTWH(cx - barW / 2, _padT + chartH - barH, barW, barH),
              );
            }

            return Stack(
              clipBehavior: Clip.none,
              children: [
                // ① Grid + Y-axis labels
                CustomPaint(
                  size: Size(totalW, totalH),
                  painter: _BarChartGridPainter(
                    padL: _padL,
                    padR: _padR,
                    padT: _padT,
                    padB: _padB,
                    chartMax: chartMax,
                    chartMin: chartMin,
                    yTickStep: yTickStep,
                    yLabelDecimals: yLabelDecimals,
                  ),
                ),

                // ② Bars
                CustomPaint(
                  size: Size(totalW, totalH),
                  painter: _BarChartBarsPainter(
                    entries: entries,
                    barRects: barRects,
                    hoveredIndex: _hoveredIndex,
                    growProg: _growAnim.value,
                    padT: _padT,
                    chartH: chartH,
                  ),
                ),

                // ③ Hover hit targets
                for (int i = 0; i < n; i++)
                  Positioned(
                    left: _padL + slotW * i,
                    top: _padT,
                    width: slotW,
                    height: chartH + _padB,
                    child: MouseRegion(
                      cursor: SystemMouseCursors.click,
                      onEnter: (_) => setState(() => _hoveredIndex = i),
                      onExit: (_) => setState(() => _hoveredIndex = null),
                      child: const SizedBox.expand(),
                    ),
                  ),

                // ④ Score value labels above each bar
                for (int i = 0; i < n; i++)
                  _buildValueLabel(entries[i], barRects[i]),

                // ⑤ Green improvement badges
                // for (int i = 0; i < n; i++)
                //   if (entries[i].improvementPct != null)
                //     _buildImprovementBadge(entries[i], barRects[i]),

                // ⑥ X-axis labels
                for (int i = 0; i < n; i++)
                  _buildXLabel(entries[i], barRects[i], totalH),

                // ⑦ Hover tooltip
                // if (_hoveredIndex != null)
                //   _buildTooltip(
                //     entries[_hoveredIndex!],
                //     barRects[_hoveredIndex!],
                //     totalW,
                //   ),
              ],
            );
          },
        );
      },
    );
  }

  // ── Score value label above bar top ───────────────────────────────────────

  Widget _buildValueLabel(_BarEntry e, Rect bar) {
    final label = _formatLikeTable(e.score);
    final hovered =
        _hoveredIndex != null && widget.entries.indexOf(e) == _hoveredIndex;
    return Positioned(
      left: bar.left - 16,
      top: bar.top - 20,
      width: bar.width + 32,
      child: Opacity(
        opacity: (_fadeAnim.value * 2).clamp(0.0, 1.0),
        child: Center(
          child: Text(
            label,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 10,
              fontWeight: hovered ? FontWeight.w700 : FontWeight.w500,
              color: hovered ? e.color : AppColors.textSecondary,
            ),
          ),
        ),
      ),
    );
  }

  // ── Improvement badge ─────────────────────────────────────────────────────

  Widget _buildImprovementBadge(_BarEntry e, Rect bar) {
    return Positioned(
      left: bar.left - 10,
      top: bar.top - 50,
      width: bar.width + 20,
      child: AnimatedOpacity(
        opacity: (_fadeAnim.value * 2).clamp(0.0, 1.0),
        duration: const Duration(milliseconds: 200),
        child: Center(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 3),
            decoration: BoxDecoration(
              color: AppColors.green.withOpacity(0.13),
              border: Border.all(color: AppColors.green.withOpacity(0.45)),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              '↓${e.improvementPct!.toStringAsFixed(1)}%',
              style: GoogleFonts.jetBrainsMono(
                fontSize: 9,
                fontWeight: FontWeight.w700,
                color: AppColors.green,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );
  }

  // ── X-axis label ──────────────────────────────────────────────────────────

  Widget _buildXLabel(_BarEntry e, Rect bar, double totalH) {
    final hovered =
        _hoveredIndex != null && widget.entries.indexOf(e) == _hoveredIndex;
    return Positioned(
      left: bar.left - 16,
      top: totalH - _padB + 10,
      width: bar.width + 32,
      child: Opacity(
        opacity: _fadeAnim.value.clamp(0.0, 1.0),
        child: Column(
          children: [
            Container(
              width: 1,
              height: 5,
              color: e.color.withOpacity(0.5),
              margin: const EdgeInsets.only(bottom: 3),
            ),
            Text(
              e.name.replaceAll('RouteIQ ', 'RL\n').replaceAll('HGS ', 'HGS\n'),
              style: AppTextStyles.monoLabel.copyWith(
                fontSize: 8,
                color: hovered ? e.color : AppColors.textSecondary,
                fontWeight: hovered ? FontWeight.w700 : FontWeight.w400,
              ),
              textAlign: TextAlign.center,
              maxLines: 2,
            ),
          ],
        ),
      ),
    );
  }

  // ── Hover tooltip ─────────────────────────────────────────────────────────

  Widget _buildTooltip(_BarEntry e, Rect bar, double totalW) {
    final label = _formatLikeTable(e.score);
    final left = (bar.center.dx - 36).clamp(0.0, totalW - 72);
    return Positioned(
      left: left,
      top: bar.top - 46,
      child: AnimatedOpacity(
        opacity: _fadeAnim.value.clamp(0.0, 1.0),
        duration: const Duration(milliseconds: 120),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
          decoration: BoxDecoration(
            color: AppColors.bg3,
            border: Border.all(color: e.color.withOpacity(0.6)),
            borderRadius: BorderRadius.circular(5),
            boxShadow: [
              BoxShadow(color: e.color.withOpacity(0.15), blurRadius: 8),
            ],
          ),
          child: Text(
            label,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 10,
              fontWeight: FontWeight.w700,
              color: e.color,
            ),
          ),
        ),
      ),
    );
  }
}

// ─── BAR CHART: GRID PAINTER ─────────────────────────────────────────────────

class _BarChartGridPainter extends CustomPainter {
  final double padL, padR, padT, padB;
  final double chartMax, chartMin;
  final double yTickStep;
  final int yLabelDecimals;

  const _BarChartGridPainter({
    required this.padL,
    required this.padR,
    required this.padT,
    required this.padB,
    required this.chartMax,
    required this.chartMin,
    required this.yTickStep,
    required this.yLabelDecimals,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = AppColors.border.withOpacity(0.45)
      ..strokeWidth = 0.5;

    final labelStyle = GoogleFonts.jetBrainsMono(
      fontSize: 9,
      color: AppColors.textMuted,
    );

    final range = chartMax - chartMin;
    if (range <= 0) return;

    final tickCount = ((chartMax - chartMin) / yTickStep).round();
    for (int i = 0; i <= tickCount; i++) {
      final val = chartMin + i * yTickStep;
      final frac = (chartMax - val) / range;
      final y = padT + frac * (size.height - padT - padB);
      canvas.drawLine(Offset(padL, y), Offset(size.width - padR, y), gridPaint);

      final label = val.toStringAsFixed(yLabelDecimals);

      final tp = TextPainter(
        text: TextSpan(text: label, style: labelStyle),
        textDirection: ui.TextDirection.ltr,
      )..layout(maxWidth: padL - 6);
      tp.paint(canvas, Offset(padL - tp.width - 4, y - tp.height / 2));
    }

    // Y-axis spine
    canvas.drawLine(
      Offset(padL, padT),
      Offset(padL, size.height - padB),
      Paint()
        ..color = AppColors.border.withOpacity(0.6)
        ..strokeWidth = 1,
    );
  }

  @override
  bool shouldRepaint(covariant _BarChartGridPainter old) =>
      old.chartMax != chartMax || old.chartMin != chartMin;
}

// ─── BAR CHART: BARS PAINTER ─────────────────────────────────────────────────

class _BarChartBarsPainter extends CustomPainter {
  final List<_BarEntry> entries;
  final List<Rect> barRects;
  final int? hoveredIndex;
  final double growProg;
  final double padT;
  final double chartH;

  const _BarChartBarsPainter({
    required this.entries,
    required this.barRects,
    required this.hoveredIndex,
    required this.growProg,
    required this.padT,
    required this.chartH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = 0; i < entries.length; i++) {
      final e = entries[i];
      final rect = barRects[i];
      if (rect.isEmpty) continue;

      final hovered = hoveredIndex == i;
      final rr = RRect.fromRectAndCorners(
        rect,
        topLeft: const Radius.circular(4),
        topRight: const Radius.circular(4),
      );

      // Bar fill with vertical gradient
      canvas.drawRRect(
        rr,
        Paint()
          ..shader = ui.Gradient.linear(
            Offset(rect.left, rect.top),
            Offset(rect.left, rect.bottom),
            [
              e.color.withOpacity(hovered ? 0.85 : 0.70),
              e.color.withOpacity(hovered ? 0.45 : 0.28),
            ],
          ),
      );

      // Border / glow on hover
      if (hovered) {
        canvas.drawRRect(
          rr,
          Paint()
            ..color = e.color.withOpacity(0.70)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BarChartBarsPainter old) =>
      old.growProg != growProg || old.hoveredIndex != hoveredIndex;
}

// ─── SHARED WIDGETS ───────────────────────────────────────────────────────────

class _StateIcon extends StatelessWidget {
  final IconData icon;
  final Color color;
  const _StateIcon({required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 64,
      height: 64,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: AppColors.bg2,
        border: Border.all(
          color: color == AppColors.textMuted
              ? AppColors.border
              : color.withOpacity(0.3),
          width: 1.5,
        ),
      ),
      child: Icon(icon, size: 28, color: color),
    );
  }
}

// ── KPI Card ──────────────────────────────────────────────────────────────────

class _BenchKpiCard extends StatefulWidget {
  final String label, value, sub;
  final Color color;
  final Color? borderColor;
  final String? tag;
  final Color? tagColor;

  const _BenchKpiCard({
    required this.label,
    required this.value,
    required this.sub,
    required this.color,
    this.borderColor,
    this.tag,
    this.tagColor,
  });

  @override
  State<_BenchKpiCard> createState() => _BenchKpiCardState();
}

class _BenchKpiCardState extends State<_BenchKpiCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        height: 140,
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
        decoration: BoxDecoration(
          color: AppColors.bg2,
          border: Border.all(
            color: _hovered
                ? widget.color.withOpacity(0.35)
                : (widget.borderColor?.withOpacity(0.27) ?? AppColors.border),
          ),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              widget.label,
              style: AppTextStyles.monoLabel,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 8),
            Text(widget.value, style: AppTextStyles.kpiValue(widget.color)),
            const SizedBox(height: 4),
            Text(
              widget.sub,
              style: AppTextStyles.monoSmall,
              overflow: TextOverflow.ellipsis,
            ),
            if (widget.tag != null) ...[
              const SizedBox(height: 8),
              _InlineTag(
                label: widget.tag!,
                color: widget.tagColor ?? widget.color,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ── Table Row ─────────────────────────────────────────────────────────────────

class _TableRow extends StatelessWidget {
  final List<String> cells;
  final bool isHeader;
  final Color? accentColor;
  final bool isBest;

  const _TableRow({
    required this.cells,
    this.isHeader = false,
    this.accentColor,
    this.isBest = false,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        Container(
          margin: EdgeInsets.only(bottom: 6, top: isBest ? 6 : 0),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          decoration: BoxDecoration(
            color: isHeader
                ? Colors.transparent
                : isBest
                ? (accentColor ?? AppColors.cyan).withOpacity(0.06)
                : AppColors.bg3,
            borderRadius: BorderRadius.circular(6),
            border: isBest
                ? Border.all(
                    color: (accentColor ?? AppColors.cyan).withOpacity(0.25),
                  )
                : null,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: cells.asMap().entries.map((entry) {
              final i = entry.key;
              final cell = entry.value;

              Color fwColor = isHeader
                  ? AppColors.textMuted
                  : AppColors.textPrimary;
              FontWeight fw = FontWeight.normal;

              if (!isHeader && i == 0 && accentColor != null) {
                fwColor = accentColor!;
                fw = FontWeight.w600;
              }
              if (!isHeader && i == 1 && isBest) {
                fwColor = accentColor ?? AppColors.green;
                fw = FontWeight.w700;
              }

              return Expanded(
                flex: i == 0 ? 2 : 1,
                child: Text(
                  cell,
                  style: isHeader
                      ? AppTextStyles.monoLabel.copyWith(fontSize: 9)
                      : AppTextStyles.monoSmall.copyWith(
                          color: fwColor,
                          fontWeight: fw,
                          fontSize: 11,
                        ),
                  overflow: TextOverflow.ellipsis,
                ),
              );
            }).toList(),
          ),
        ),
        if (isBest)
          Positioned(
            top: -1,
            left: 10,
            child: Container(
              color: AppColors.bg2,
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Text(
                'BEST',
                style: AppTextStyles.monoLabel.copyWith(
                  fontSize: 9,
                  color: accentColor ?? AppColors.green,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
      ],
    );
  }
}

// ── Bench Card ────────────────────────────────────────────────────────────────

class _BenchCard extends StatelessWidget {
  final Widget child;
  const _BenchCard({required this.child});

  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
    decoration: BoxDecoration(
      color: AppColors.bg2,
      borderRadius: BorderRadius.circular(10),
      border: Border.all(color: AppColors.border),
    ),
    child: child,
  );
}

// ── Outline Button ────────────────────────────────────────────────────────────

class _OutlineButton extends StatefulWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback? onTap;
  final bool fullWidth;

  const _OutlineButton({
    required this.label,
    required this.icon,
    required this.color,
    this.onTap,
    this.fullWidth = false,
  });

  @override
  State<_OutlineButton> createState() => _OutlineButtonState();
}

class _OutlineButtonState extends State<_OutlineButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: widget.onTap == null
          ? SystemMouseCursors.forbidden
          : SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: widget.fullWidth ? double.infinity : null,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          decoration: BoxDecoration(
            color: _hovered && widget.onTap != null
                ? widget.color.withOpacity(0.12)
                : widget.color.withOpacity(0.06),
            border: Border.all(
              color: widget.onTap == null
                  ? AppColors.border
                  : widget.color.withOpacity(0.5),
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisSize: widget.fullWidth
                ? MainAxisSize.max
                : MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                widget.icon,
                size: 15,
                color: widget.onTap == null
                    ? AppColors.textMuted
                    : widget.color,
              ),
              const SizedBox(width: 8),
              Text(
                widget.label,
                style: GoogleFonts.syne(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.06,
                  color: widget.onTap == null
                      ? AppColors.textMuted
                      : widget.color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Legend Dot ────────────────────────────────────────────────────────────────

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 5),
        Text(
          label,
          style: AppTextStyles.monoSmall.copyWith(
            color: AppColors.textSecondary,
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}

// ── Inline Tag ────────────────────────────────────────────────────────────────

class _InlineTag extends StatelessWidget {
  final String label;
  final Color color;
  const _InlineTag({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        border: Border.all(color: color.withOpacity(0.4)),
        borderRadius: BorderRadius.circular(3),
      ),
      child: Text(
        label,
        style: GoogleFonts.jetBrainsMono(
          fontSize: 9,
          fontWeight: FontWeight.w500,
          color: color,
          letterSpacing: 0.08,
        ),
      ),
    );
  }
}

// ── Grid Background ───────────────────────────────────────────────────────────

class _GridBackground extends StatelessWidget {
  const _GridBackground();

  @override
  Widget build(BuildContext context) =>
      Positioned.fill(child: CustomPaint(painter: _GridPainter()));
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = AppColors.border.withOpacity(0.3)
      ..strokeWidth = 0.5;
    const s = 40.0;
    for (double x = 0; x < size.width; x += s) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), p);
    }
    for (double y = 0; y < size.height; y += s) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), p);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}
