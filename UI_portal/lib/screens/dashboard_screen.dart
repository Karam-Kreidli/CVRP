import 'package:RouteIQ_UI/services/solver_controller.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/utils/http_utils.dart';
import 'package:RouteIQ_UI/utils/logger.dart';
import 'package:RouteIQ_UI/widgets/Tag.dart';
import 'package:RouteIQ_UI/widgets/animatedDot.dart';
import 'package:RouteIQ_UI/widgets/sectionlabel_line.dart';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';

import 'dart:async';
import 'dart:convert';
import 'dart:math' show min;
import 'package:http/http.dart' as http;

// ─── API ─────────────────────────────────────────────────────────────────────
const _kBaseUrl = ApiConfig.baseUrl;
// GET /runs?limit=20 → { total, runs: [...] }
// NOTE: Dashboard no longer calls GET /health itself.
//       HealthState is passed in from AppShell — the single source of truth.
//       Data is only fetched when health.isReady == true.
// ─────────────────────────────────────────────────────────────────────────────

// ─── EMPTY/DASH CONSTANTS ────────────────────────────────────────────────────
const _kDash = '—';
List<double> _kZeroTrend = const [
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
];

// ─── PIPELINE STAGE MODEL ────────────────────────────────────────────────────

class _PipelineStage {
  final String name;
  final double pct;
  final String status;
  final Color color;

  const _PipelineStage({
    required this.name,
    required this.pct,
    required this.status,
    required this.color,
  });

  factory _PipelineStage.fromPct(String name, double pct) {
    final String status;
    final Color color;
    if (pct >= 1.0) {
      status = 'Active';
      color = AppColors.green;
    } else if (pct >= 0.75) {
      status = 'Active';
      color = AppColors.amber;
    } else {
      status = 'Training';
      color = AppColors.cyan;
    }
    return _PipelineStage(name: name, pct: pct, status: status, color: color);
  }

  factory _PipelineStage.empty(String name) => _PipelineStage(
    name: name,
    pct: 0,
    status: _kDash,
    color: AppColors.textMuted,
  );
}

// ─── RUN RECORD MODEL ────────────────────────────────────────────────────────

class _RunRecord {
  final String name;
  final int nv;
  final String td;
  final String score;
  final String time;
  final bool running;

  const _RunRecord({
    required this.name,
    required this.nv,
    required this.td,
    required this.score,
    required this.time,
    this.running = false,
  });

  factory _RunRecord.fromJson(Map<String, dynamic> j) => _RunRecord(
    name: j['instance_name'] as String,
    nv: j['num_vehicles'] as int,
    td: (j['total_distance'] as num).toStringAsFixed(1),
    score: (j['score'] as num).toStringAsFixed(1),
    time: '${j['solve_time_seconds']}s',
    running: j['status'] == 'running',
  );
}

// ─── EMPTY STATE HELPERS ──────────────────────────────────────────────────────

// ── CHANGE: 4 stages replacing the old 5 ────────────────────────────────────
//
// Removed: GNN Observer, Route Driver, MACA Trainer
// Added:   Feature Extractor (stage 1), PPO Trainer (stage 4)
//
// Stage names here must match the keys returned in GET /health → stage_health:
//   feature_extractor, fleet_manager, hgs_engine, ppo_trainer

List<_PipelineStage> _emptyStages() => [
  _PipelineStage.empty('Feature Extractor'),
  _PipelineStage.empty('Fleet Manager'),
  _PipelineStage.empty('HGS Engine'),
  _PipelineStage.empty('PPO Trainer'),
];

// ─── DASHBOARD SCREEN ────────────────────────────────────────────────────────

class DashboardScreen extends StatefulWidget {
  /// HealthState from AppShell — the single source of truth for connectivity.
  /// Dashboard only fetches data when health.isReady == true.
  final HealthState health;

  const DashboardScreen({super.key, required this.health});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // KPI cards
  String _bestNv = '00';
  String _bestTd = '0000';
  String _bestScore = '0,000';
  String _runsToday = '0';

  // ── Table — empty until backend is ready ──────────────────────────────────
  List<_RunRecord> _runs = const [];

  // ── Chart — empty until backend is ready ──────────────────────────────────
  List<double> _trendPoints = _kZeroTrend;

  // ── Pipeline health — empty bars until backend confirms status ────────────
  List<_PipelineStage> _stages = _emptyStages();

  // ── Internal ──────────────────────────────────────────────────────────────
  Timer? _refreshTimer;
  bool _hasFetchedOnce = false;

  @override
  void initState() {
    super.initState();
    // Listen to SolverController so we can reflect live solving in the table
    SolverController.instance.addListener(_onSolverChanged);
    // Attempt first data load — will no-op if health is not ready yet
    _maybeLoadData();
  }

  @override
  void didUpdateWidget(DashboardScreen oldWidget) {
    super.didUpdateWidget(oldWidget);

    final wasReady = oldWidget.health.isReady;
    final isReady = widget.health.isReady;

    if (!wasReady && isReady) {
      _loadData();
    }

    if (wasReady && !isReady) {
      _refreshTimer?.cancel();
      _clearToEmpty();
    }
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    SolverController.instance.removeListener(_onSolverChanged);
    super.dispose();
  }

  // ── SolverController listener — rebuilds when solver status changes ────────
  // This makes the "RUNNING" row appear/disappear in the recent runs table
  // the moment a solve starts or finishes, without waiting for the next poll.
  void _onSolverChanged() {
    if (mounted) setState(() {});
  }

  // ── Gate: only load data when backend is confirmed ready ──────────────────
  void _maybeLoadData() {
    if (widget.health.isReady) {
      _loadData();
    }
  }

  // ── Reset all displayed values back to dashes ─────────────────────────────
  void _clearToEmpty() {
    if (!mounted) return;
    setState(() {
      _bestNv = _kDash;
      _bestTd = _kDash;
      _bestScore = _kDash;
      _runsToday = _kDash;
      _runs = const [];
      _trendPoints = _kZeroTrend;
      _stages = _emptyStages();
      _hasFetchedOnce = false;
    });
  }

  // ── Main data load — only called when health.isReady == true ──────────────
  Future<void> _loadData() async {
    if (!mounted) return;

    if (!widget.health.isReady) {
      _clearToEmpty();
      return;
    }

    final results = await Future.wait([
      http
          .get(Uri.parse('$_kBaseUrl/runs?limit=20'))
          .catchError((_) => http.Response('', 0)),
      http
          .get(Uri.parse('$_kBaseUrl/health'))
          .catchError((_) => http.Response('', 0)),
    ]);

    if (!mounted) return;

    final runsRes = results[0];
    final healthRes = results[1];

    // ── /runs response ────────────────────────────────────────────────────────
    if (runsRes.statusCode == 200) {
      try {
        final body = jsonDecode(runsRes.body) as Map<String, dynamic>;
        final rawRuns = (body['runs'] as List).cast<Map<String, dynamic>>();
        final loaded = rawRuns.map(_RunRecord.fromJson).toList();
        final completed = loaded.where((r) => !r.running).toList();

        final allNv = completed.map((r) => r.nv).toList();
        final allTd = completed
            .map((r) => double.tryParse(r.td.replaceAll(',', '')) ?? 0.0)
            .toList();
        final allScore = completed
            .map((r) => double.tryParse(r.score.replaceAll(',', '')) ?? 0.0)
            .toList();

        final trendScores = allScore.reversed.take(12).toList().toList();

        setState(() {
          _runs = loaded;
          _runsToday = body['total'].toString();
          _bestNv = allNv.isNotEmpty ? allNv.reduce(min).toString() : _kDash;
          _bestTd = allTd.isNotEmpty
              ? allTd.reduce(min).toStringAsFixed(0)
              : _kDash;
          _bestScore = allScore.isNotEmpty
              ? _fmtInt(allScore.reduce(min).toInt())
              : _kDash;
          if (trendScores.isNotEmpty) _trendPoints = trendScores;
          _hasFetchedOnce = true;
        });
      } catch (_) {}
    }

    // ── /health response — for pipeline health bars only ──────────────────────
    if (healthRes.statusCode == 200) {
      try {
        final h = jsonDecode(healthRes.body) as Map<String, dynamic>;
        final sh = h['stage_health'] as Map<String, dynamic>? ?? {};

        // ── CHANGE: parse 4 stage health keys matching new architecture ───────
        //
        // Old keys removed: gnn_observer, route_driver, maca_trainer
        // New keys:
        //   feature_extractor — always 1.0 (deterministic, no training)
        //   fleet_manager     — PPO convergence ratio (epoch / total epochs)
        //   hgs_engine        — always 1.0 (classical solver, not trained)
        //   ppo_trainer       — PPO training progress ratio

        final newStages = [
          _PipelineStage.fromPct(
            'Feature Extractor',
            (sh['feature_extractor'] as num?)?.toDouble() ?? 0.0,
          ),
          _PipelineStage.fromPct(
            'Fleet Manager',
            (sh['fleet_manager'] as num?)?.toDouble() ?? 0.0,
          ),
          _PipelineStage.fromPct(
            'HGS Engine',
            (sh['hgs_engine'] as num?)?.toDouble() ?? 0.0,
          ),
          _PipelineStage.fromPct(
            'PPO Trainer',
            (sh['ppo_trainer'] as num?)?.toDouble() ?? 0.0,
          ),
        ];
        if (mounted) setState(() => _stages = newStages);
      } catch (_) {}
    }

    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(
      const Duration(seconds: 30),
      (_) => _loadData(),
    );
  }

  String _fmtInt(int v) {
    final s = v.toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg0,
      child: Stack(
        children: [
          const _GridBackground(),
          SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(32, 28, 32, 40),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),
                const SizedBox(height: 28),
                if (!widget.health.isReady) _buildStatusBanner(),
                if (!widget.health.isReady) const SizedBox(height: 16),
                _buildKpiRow(context),
                const SizedBox(height: 16),
                _buildMiddleRow(context),
                const SizedBox(height: 16),
                _buildRecentRunsTable(),
                const SizedBox(height: 16),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusBanner() {
    final isLoading = widget.health.status == SystemStatus.loading;
    final color = widget.health.dotColor;
    final message = isLoading
        ? 'Connecting to backend…  Data will appear once connected.'
        : 'Backend unreachable.  Check that the server is running and the URL is correct.';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        border: Border.all(color: color.withOpacity(0.35)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          AnimatedStatusDot(color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: AppTextStyles.mono.copyWith(color: color, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Flexible(
              child: Text('Command Center', style: AppTextStyles.displayMedium),
            ),
            const SizedBox(width: 12),
            Tag(label: 'CVRP', color: AppColors.amber, fixWidth: false),
          ],
        ),
        const SizedBox(height: 6),
      ],
    );
  }

  Widget _buildKpiRow(BuildContext context) {
    final compact = MediaQuery.of(context).size.width < 1100;
    final cards = [
      _KpiCard(
        label: 'BEST VEHICLES',
        value: _bestNv,
        unit: 'trucks',
        color: AppColors.cyan,
        isReady: widget.health.isReady,
      ),
      _KpiCard(
        label: 'BEST DISTANCE',
        value: _bestTd,
        unit: 'km',
        color: AppColors.amber,
        isReady: widget.health.isReady,
      ),
      _KpiCard(
        label: 'BEST SCORE',
        value: _bestScore,
        unit: 'pts',
        color: AppColors.green,
        isReady: widget.health.isReady,
      ),
      _KpiCard(
        label: 'RUNS TODAY',
        value: _runsToday,
        unit: 'jobs',
        color: AppColors.textSecondary,
        isReady: widget.health.isReady,
      ),
    ];

    if (compact) {
      return GridView.count(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1.6,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        children: cards,
      );
    }
    return Row(
      children:
          cards
              .map((c) => Expanded(child: c))
              .expand((w) => [w, const SizedBox(width: 12)])
              .toList()
            ..removeLast(),
    );
  }

  Widget _buildMiddleRow(BuildContext context) {
    final narrow = MediaQuery.of(context).size.width < 1000;

    final chart = _SectionCard(
      title: !(widget.health.isReady)
          ? 'SCORE TREND'
          : 'SCORE TREND — LAST ${_trendPoints.length} RUNS',
      child: _ScoreTrendChart(
        points: _trendPoints,
        isReady: widget.health.isReady,
      ),
    );

    final health = _SectionCard(
      title: 'PIPELINE HEALTH',
      child: _PipelineHealthWidget(
        stages: _stages,
        isReady: widget.health.isReady,
      ),
    );

    if (narrow)
      return Column(children: [chart, const SizedBox(height: 12), health]);
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(flex: 2, child: chart),
          const SizedBox(width: 12),
          Expanded(flex: 1, child: health),
        ],
      ),
    );
  }

  // ── Recent Runs Table ─────────────────────────────────────────────────────
  //
  // If a solve is actively running, a live "RUNNING" row for that file is
  // prepended at the top of the table — regardless of what /runs returns.
  // This disappears the moment the controller transitions out of running state.

  Widget _buildRecentRunsTable() {
    final ctrl = SolverController.instance;

    // Build the display list: inject a synthetic running row if solver is live
    final List<_RunRecord> displayRuns;
    if (ctrl.status == SolverStatus.running && ctrl.instanceInfo != null) {
      final info = ctrl.instanceInfo!;
      final liveRow = _RunRecord(
        name: info.fileName,
        nv: 0,
        td: '—',
        score: '—',
        time: '—',
        running: true,
      );
      displayRuns = [liveRow, ..._runs.where((r) => !r.running)];
    } else {
      displayRuns = _runs;
    }

    return _SectionCard(
      title: 'RECENT RUNS',
      child: displayRuns.isEmpty
          ? _buildEmptyTable()
          : Column(
              children: [
                _TableRow(
                  cells: const [
                    'INSTANCE',
                    'NV',
                    'TOTAL DISTANCE',
                    'SCORE',
                    'TIME',
                    'STATUS',
                  ],
                  isHeader: true,
                ),
                const Divider(height: 1, color: AppColors.border),
                ...displayRuns.map(
                  (r) => Column(
                    children: [
                      _TableRow(run: r),
                      Divider(
                        height: 1,
                        color: AppColors.border.withOpacity(0.4),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildEmptyTable() {
    final ctrl = SolverController.instance;
    final String message;

    if (ctrl.status == SolverStatus.running && ctrl.instanceInfo != null) {
      message = 'Solving ${ctrl.instanceInfo!.fileName}…';
    } else if (widget.health.status == SystemStatus.loading) {
      message = 'Connecting to backend…';
    } else if (widget.health.status == SystemStatus.error) {
      message = 'Backend unreachable — no run history available';
    } else {
      message = 'No runs yet — upload a .vrp file and click Run Solver';
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Center(
        child: Text(
          message,
          style: AppTextStyles.mono.copyWith(color: AppColors.textMuted),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}

// ─── SECTION CARD ─────────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  const _SectionCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 0),
      decoration: BoxDecoration(
        color: AppColors.bg2,
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SectionLabel(title: title),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }
}

// ─── KPI CARD ─────────────────────────────────────────────────────────────────

class _KpiCard extends StatefulWidget {
  final String label, value, unit;
  final Color color;
  final bool isReady;

  const _KpiCard({
    required this.label,
    required this.value,
    required this.unit,
    required this.color,
    required this.isReady,
  });

  @override
  State<_KpiCard> createState() => _KpiCardState();
}

class _KpiCardState extends State<_KpiCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final displayColor = widget.isReady ? widget.color : AppColors.textMuted;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
        decoration: BoxDecoration(
          color: AppColors.bg2,
          border: Border.all(
            color: (_hovered && widget.isReady)
                ? widget.color.withOpacity(0.35)
                : AppColors.border,
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
            Flexible(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Flexible(
                    child: FittedBox(
                      fit: BoxFit.scaleDown,
                      alignment: Alignment.bottomLeft,
                      child: Text(
                        widget.value,
                        style: AppTextStyles.kpiValue(displayColor),
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(widget.unit, style: AppTextStyles.kpiUnit),
                ],
              ),
            ),
            const SizedBox(height: 4),
          ],
        ),
      ),
    );
  }
}

// ─── SCORE TREND CHART ────────────────────────────────────────────────────────

class _ScoreTrendChart extends StatelessWidget {
  final List<double> points;
  final bool isReady;
  const _ScoreTrendChart({required this.points, required this.isReady});

  @override
  Widget build(BuildContext context) {
    if (points.isEmpty) {
      return SizedBox(
        height: 160,
        child: Center(
          child: Text(
            'No run data yet',
            style: AppTextStyles.mono.copyWith(color: AppColors.textMuted),
          ),
        ),
      );
    }

    final spots = points
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    final rawMin = points.reduce(min);
    final rawMax = points.reduce((a, b) => a > b ? a : b);
    final double minY;
    final double maxY;
    if (rawMin == rawMax) {
      minY = 0;
      maxY = 100;
    } else {
      minY = (rawMin * 0.9).floorToDouble();
      maxY = (rawMax * 1.1).ceilToDouble();
    }

    final lineColor = isReady ? AppColors.cyan : AppColors.textMuted;

    return SizedBox(
      height: 220,
      child: LineChart(
        LineChartData(
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            getDrawingHorizontalLine: (_) => FlLine(
              color: AppColors.border.withOpacity(0.6),
              strokeWidth: 0.5,
            ),
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                getTitlesWidget: (v, _) =>
                    Text(v.toInt().toString(), style: AppTextStyles.monoSmall),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: 1,
                reservedSize: 40,
                getTitlesWidget: (v, _) => Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    v.toInt().toString(),
                    style: AppTextStyles.monoSmall,
                  ),
                ),
              ),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
          ),
          borderData: FlBorderData(show: false),
          minY: minY,
          maxY: maxY,
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              curveSmoothness: 0.35,
              color: lineColor,
              barWidth: 2,
              dotData: FlDotData(
                show: true,
                getDotPainter: (_, __, ___, ____) => FlDotCirclePainter(
                  radius: 3,
                  color: AppColors.bg2,
                  strokeWidth: 1.5,
                  strokeColor: lineColor,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    lineColor.withOpacity(0.15),
                    lineColor.withOpacity(0.0),
                  ],
                ),
              ),
            ),
          ],
        ),
        duration: const Duration(milliseconds: 600),
        curve: Curves.easeInOut,
      ),
    );
  }
}

// ─── PIPELINE HEALTH ─────────────────────────────────────────────────────────

class _PipelineHealthWidget extends StatelessWidget {
  final List<_PipelineStage> stages;
  final bool isReady;
  const _PipelineHealthWidget({required this.stages, required this.isReady});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: stages
          .map((s) => _StageRow(stage: s, isReady: isReady))
          .toList(),
    );
  }
}

class _StageRow extends StatelessWidget {
  final _PipelineStage stage;
  final bool isReady;
  const _StageRow({required this.stage, required this.isReady});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                child: Text(
                  stage.name,
                  style: AppTextStyles.mono.copyWith(
                    color: AppColors.textSecondary,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                stage.status,
                style: AppTextStyles.monoSmall.copyWith(color: stage.color),
              ),
            ],
          ),
          const SizedBox(height: 5),
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: stage.pct,
              minHeight: 3,
              backgroundColor: AppColors.border,
              valueColor: AlwaysStoppedAnimation<Color>(stage.color),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── TABLE ROW ────────────────────────────────────────────────────────────────

class _TableRow extends StatefulWidget {
  final List<String>? cells;
  final _RunRecord? run;
  final bool isHeader;
  const _TableRow({this.cells, this.run, this.isHeader = false});

  @override
  State<_TableRow> createState() => _TableRowState();
}

class _TableRowState extends State<_TableRow> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    if (widget.isHeader) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 10),
        child: Row(
          children: widget.cells!
              .asMap()
              .entries
              .map(
                (e) => Expanded(
                  flex: _flex(e.key),
                  child: Text(
                    e.value,
                    style: AppTextStyles.tableHeader.copyWith(
                      letterSpacing: 0.1,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              )
              .toList(),
        ),
      );
    }

    final r = widget.run!;
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(6),
          color: _hovered ? AppColors.cyanFaint : Colors.transparent,
        ),
        child: Row(
          children: [
            Expanded(
              flex: _flex(0),
              child: Text(
                r.name,
                style: AppTextStyles.tableCell.copyWith(
                  color: AppColors.textPrimary,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Expanded(
              flex: _flex(1),
              child: Text(
                r.running ? '—' : r.nv.toString(),
                style: AppTextStyles.tableCellColored(AppColors.cyan),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Expanded(
              flex: _flex(2),
              child: Text(
                r.td,
                style: AppTextStyles.tableCell,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Expanded(
              flex: _flex(3),
              child: Text(
                r.score,
                style: AppTextStyles.tableCellColored(AppColors.amber),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Expanded(
              flex: _flex(4),
              child: Text(
                r.time,
                style: AppTextStyles.tableCell.copyWith(
                  color: AppColors.textMuted,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Flexible(
              flex: _flex(5),
              child: Tag(
                label: r.running ? 'RUNNING' : 'DONE',
                color: r.running ? AppColors.cyan : AppColors.green,
                fixWidth: true,
              ),
            ),
          ],
        ),
      ),
    );
  }

  int _flex(int col) => const [3, 1, 2, 2, 2, 2][col];
}

// ─── GRID BACKGROUND ─────────────────────────────────────────────────────────

class _GridBackground extends StatelessWidget {
  const _GridBackground();
  @override
  Widget build(BuildContext context) =>
      Positioned.fill(child: CustomPaint(painter: _GridPainter()));
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.border.withOpacity(0.3)
      ..strokeWidth = 0.5;
    const spacing = 40.0;
    for (double x = 0; x < size.width; x += spacing) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += spacing) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}
