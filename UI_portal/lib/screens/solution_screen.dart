// ignore_for_file: unused_import
//new
import 'dart:async';
import 'dart:math' as math;
import 'dart:convert';
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
import 'package:RouteIQ_UI/services/solver_controller.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/http_utils.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'package:http/http.dart' as http;

// ─── API INTEGRATION ─────────────────────────────────────────────────────────
//
const _kBaseUrl = ApiConfig.baseUrl;
//
// Endpoint: GET /result/{job_id}
// Returns:
// {
//   "job_id": string,
//   "instance_name": string,
//   "num_nodes": int,
//   "num_vehicles": int,
//   "total_distance": double,
//   "score": double,
//   "nv_min": int,
//   "solve_time_seconds": int,
//   "depot": { "id": int, "x": double, "y": double },
//   "customers": [ { "id": int, "x": double, "y": double, "demand": int } ],
//   "routes": [
//     {
//       "route_id": int,
//       "customer_ids": [int],   ← CVRPLIB convention: 1 … n-1
//       "num_stops": int,
//       "distance": double,
//       "total_load": int,
//       "capacity": int,
//       "capacity_pct": double
//     }
//   ]
// }
//
// NOTE: customer_ids are already in CVRPLIB convention (1-based, depot excluded).
//       The TSPLIB95 node index for customer k  =  k + 1  (depot = 1).
//
// ─────────────────────────────────────────────────────────────────────────────

// ─── MODELS ───────────────────────────────────────────────────────────────────

class _NodePoint {
  final int id;
  final double x;
  final double y;
  final int demand;
  const _NodePoint({
    required this.id,
    required this.x,
    required this.y,
    this.demand = 0,
  });
}

class _RouteData {
  final int routeId;
  final List<int> customerIds;
  final int numStops;
  final double distance;
  final int totalLoad;
  final int capacity;
  final double capacityPct;
  const _RouteData({
    required this.routeId,
    required this.customerIds,
    required this.numStops,
    required this.distance,
    required this.totalLoad,
    required this.capacity,
    required this.capacityPct,
  });

  factory _RouteData.fromJson(Map<String, dynamic> j) => _RouteData(
    routeId: j['route_id'] as int,
    customerIds: (j['customer_ids'] as List).cast<int>(),
    numStops: j['num_stops'] as int,
    distance: (j['distance'] as num).toDouble(),
    totalLoad: j['total_load'] as int,
    capacity: j['capacity'] as int,
    capacityPct: (j['capacity_pct'] as num).toDouble(),
  );
}

class _SolutionData {
  final String instanceName;
  final int numVehicles;
  final double totalDistance;
  final double score;
  final int solveTimeSeconds;
  final _NodePoint depot;
  final List<_NodePoint> customers;
  final List<_RouteData> routes;

  const _SolutionData({
    required this.instanceName,
    required this.numVehicles,
    required this.totalDistance,
    required this.score,
    required this.solveTimeSeconds,
    required this.depot,
    required this.customers,
    required this.routes,
  });

  factory _SolutionData.fromJson(Map<String, dynamic> j) => _SolutionData(
    instanceName: j['instance_name'] as String,
    numVehicles: j['num_vehicles'] as int,
    totalDistance: (j['total_distance'] as num).toDouble(),
    score: (j['score'] as num).toDouble(),
    solveTimeSeconds: j['solve_time_seconds'] as int,
    depot: _NodePoint(
      id: j['depot']['id'],
      x: j['depot']['x'].toDouble(),
      y: j['depot']['y'].toDouble(),
    ),
    customers: (j['customers'] as List)
        .map(
          (c) => _NodePoint(
            id: c['id'],
            x: c['x'].toDouble(),
            y: c['y'].toDouble(),
            demand: c['demand'],
          ),
        )
        .toList(),
    routes: (j['routes'] as List).map((r) => _RouteData.fromJson(r)).toList(),
  );
}

// ── Empty mock shown when no jobId is provided ────────────────────────────────

_SolutionData _buildMock() => const _SolutionData(
  instanceName: '—',
  numVehicles: 0,
  totalDistance: 0.0,
  score: 0.0,
  solveTimeSeconds: 0,
  depot: _NodePoint(id: 0, x: 50, y: 50),
  customers: [],
  routes: [],
);

final _mockSolution = _buildMock();

// ─── SOLUTION VIEWER SCREEN ───────────────────────────────────────────────────

class SolutionScreen extends StatefulWidget {
  /// Pass a jobId when navigating here after a solve completes.
  /// Leave null to show mock / empty data.
  final String? jobId;

  const SolutionScreen({super.key, this.jobId});

  @override
  State<SolutionScreen> createState() => _SolutionScreenState();
}

class _SolutionScreenState extends State<SolutionScreen> {
  // Max poll attempts for HTTP 425 (job still running).
  // 30 attempts × 2 s = 60 s maximum wait before showing an error.
  static const _kMaxRetries = 30;

  _SolutionData? _solution;
  bool _loading = false;
  String? _error;
  int? _hoveredRouteId;

  @override
  void initState() {
    super.initState();
    _loadSolution();
  }

  Future<void> _loadSolution() async {
    final ctrl = SolverController.instance;

    // If no jobId provided, try to use cached solution from controller
    if (widget.jobId == null) {
      if (ctrl.cachedSolutionJson != null) {
        setState(
          () => _solution = _SolutionData.fromJson(ctrl.cachedSolutionJson!),
        );
      } else {
        setState(() => _solution = _mockSolution);
      }
      return;
    }

    // If the requested jobId matches cached data, use it immediately
    if (widget.jobId == ctrl.cachedSolutionJobId &&
        ctrl.cachedSolutionJson != null) {
      setState(
        () => _solution = _SolutionData.fromJson(ctrl.cachedSolutionJson!),
      );
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    final url = '$_kBaseUrl/result/${widget.jobId}';

    for (int attempt = 0; attempt < _kMaxRetries; attempt++) {
      try {
        final res = await http
            .get(Uri.parse(url))
            .timeout(const Duration(seconds: 10));

        if (!mounted) return;

        if (res.statusCode == 200) {
          final body = jsonDecode(res.body) as Map<String, dynamic>;
          ctrl.cacheSolution(widget.jobId!, body);
          setState(() {
            _solution = _SolutionData.fromJson(body);
            _loading = false;
          });
          return;
        } else if (res.statusCode == 425) {
          await Future.delayed(const Duration(seconds: 2));
          continue;
        } else {
          setState(() {
            _error = 'Failed to load result (${res.statusCode})';
            _loading = false;
          });
          return;
        }
      } on TimeoutException {
        if (!mounted) return;
        await Future.delayed(const Duration(seconds: 2));
        continue;
      } catch (e) {
        if (!mounted) return;
        setState(() {
          _error = 'Network error: $e';
          _loading = false;
        });
        return;
      }
    }

    if (mounted) {
      setState(() {
        _error = 'Solver is taking too long — please retry.';
        _loading = false;
      });
    }
  }

  // ── Export ────────────────────────────────────────────────────────────────

  /// Builds a CVRPLIB-format solution string and triggers a browser download.
  ///
  /// Output format (DIMACS / CVRPLIB convention):
  ///   Route #1: 3 1 2
  ///   Route #2: 6 5 4
  ///
  /// customer_ids from the API are already in CVRPLIB numbering (1 … n-1),
  /// so they are written as-is — no offset adjustment needed.
  void _exportSol() {
    final sol = _solution;
    if (sol == null || sol.customers.isEmpty) return;

    // ── Build the text content ──────────────────────────────────────────────
    final buffer = StringBuffer();
    for (final route in sol.routes) {
      buffer.write('Route #${route.routeId}: ');
      buffer.writeln(route.customerIds.join(' '));
    }
    final content = buffer.toString();

    // ── Trigger browser download ────────────────────────────────────────────
    final bytes = utf8.encode(content);
    final blob = html.Blob([bytes], 'text/plain');
    final url = html.Url.createObjectUrlFromBlob(blob);

    // Sanitise instance name for use as a filename
    final safeName = sol.instanceName.replaceAll(RegExp(r'[^\w\-.]'), '_');
    final filename = '${safeName}_solution.txt';

    final anchor = html.document.createElement('a') as html.AnchorElement
      ..href = url
      ..setAttribute('download', filename);
    html.document.body!.append(anchor);
    anchor.click();
    anchor.remove();
    html.Url.revokeObjectUrl(url);

    _showExportToast('Downloaded $filename');
  }

  void _showExportToast(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          msg,
          style: AppTextStyles.subheading.copyWith(
            color: AppColors.textPrimary,
          ),
        ),
        backgroundColor: AppColors.green,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(6),
          side: const BorderSide(color: AppColors.border),
        ),
        duration: const Duration(seconds: 3),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg0,
      child: Stack(
        children: [
          const _GridBackground(),
          if (_loading)
            const Center(
              child: CircularProgressIndicator(color: AppColors.cyan),
            )
          else if (_error != null)
            _buildError()
          else if (_solution != null)
            _buildContent(_solution!)
          else
            _buildEmpty(),
        ],
      ),
    );
  }

  Widget _buildError() => Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text('◈', style: TextStyle(fontSize: 40, color: AppColors.red)),
        const SizedBox(height: 12),
        Text(_error!, style: AppTextStyles.mono.copyWith(color: AppColors.red)),
        const SizedBox(height: 16),
        _ExportButton(
          icon: Icons.refresh,
          label: 'Retry',
          color: AppColors.cyan,
          onTap: _loadSolution,
        ),
      ],
    ),
  );

  Widget _buildEmpty() => Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          '◈',
          style: TextStyle(fontSize: 48, color: AppColors.textMuted),
        ),
        const SizedBox(height: 16),
        Text(
          'No solution loaded',
          style: AppTextStyles.displayMedium.copyWith(fontSize: 18),
        ),
        const SizedBox(height: 6),
        Text(
          'Run the solver first, then come back here.',
          style: AppTextStyles.mono,
        ),
      ],
    ),
  );

  Widget _buildContent(_SolutionData sol) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(32, 28, 32, 40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(sol),
          const SizedBox(height: 20),
          _buildScoreBanner(sol),
          const SizedBox(height: 16),
          _buildMainRow(context, sol),
        ],
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _buildHeader(_SolutionData sol) {
    final mins = sol.solveTimeSeconds ~/ 60;
    final secs = sol.solveTimeSeconds % 60;
    final timeStr = mins > 0
        ? '${mins}m ${secs.toString().padLeft(2, '0')}s'
        : '${secs}s';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Solution Viewer', style: AppTextStyles.displayMedium),
        const SizedBox(height: 6),
        Text(
          '${sol.instanceName} · Solved in $timeStr',
          style: AppTextStyles.mono,
        ),
      ],
    );
  }

  // ── Score Banner ──────────────────────────────────────────────────────────

  Widget _buildScoreBanner(_SolutionData sol) {
    final baselineScore = sol.score / (1 - 0.184);
    final improvement = ((baselineScore - sol.score) / baselineScore * 100);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
          colors: [AppColors.cyanFaint, AppColors.bg2],
        ),
        border: Border.all(color: AppColors.cyan.withOpacity(0.25)),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          _BannerKpi(
            label: 'VEHICLES',
            value: sol.numVehicles.toString(),
            unit: '',
            color: AppColors.cyan,
            solution: sol,
          ),
          _BannerDivider(),
          _BannerKpi(
            label: 'TOTAL DISTANCE',
            value: _fmt(sol.totalDistance),
            unit: 'km',
            color: AppColors.amber,
            solution: sol,
          ),
          _BannerDivider(),
          _BannerKpi(
            label: 'FINAL SCORE',
            value: _fmt(sol.score),
            unit: '',
            color: AppColors.green,
            solution: sol,
          ),
          _BannerDivider(),
          _BannerKpi(
            label: 'IMPROVEMENT',
            value: improvement.toStringAsFixed(1),
            unit: '%',
            color: AppColors.purple,
            solution: sol,
          ),
          _BannerDivider(),
          _BannerKpi(
            label: 'SOLVE TIME',
            value: sol.solveTimeSeconds.toString(),
            unit: 's',
            color: AppColors.textSecondary,
            solution: sol,
          ),
        ],
      ),
    );
  }

  // ── Main Row ──────────────────────────────────────────────────────────────

  Widget _buildMainRow(BuildContext context, _SolutionData sol) {
    final narrow = MediaQuery.of(context).size.width < 1050;

    final canvas = _SectionCard(
      title: 'ROUTE MAP',
      child: _RouteCanvas(
        solution: sol,
        hoveredRouteId: _hoveredRouteId,
        onHover: (id) => setState(() => _hoveredRouteId = id),
      ),
    );

    final panel = Column(
      children: [
        _buildExportPanel(),
        const SizedBox(height: 14),
        _buildRouteTable(sol),
      ],
    );

    if (narrow) {
      return Column(children: [canvas, const SizedBox(height: 14), panel]);
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(flex: 3, child: canvas),
        const SizedBox(width: 14),
        SizedBox(width: 300, child: panel),
      ],
    );
  }

  // ── Export Panel ──────────────────────────────────────────────────────────

  Widget _buildExportPanel() {
    final hasNoData = _solution?.customers.isEmpty ?? true;

    return _SectionCard(
      title: 'EXPORT',
      child: Column(
        children: [
          _ExportButton(
            icon: Icons.description_outlined,
            label: 'Export .txt',
            color: AppColors.cyan,
            onTap: _exportSol,
            disabled: hasNoData,
          ),
        ],
      ),
    );
  }

  // ── Route Table ───────────────────────────────────────────────────────────

  Widget _buildRouteTable(_SolutionData sol) {
    return _SectionCard(
      title: 'ROUTES',
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.only(bottom: 8, right: 8, left: 8),
            child: Row(
              children: [
                const SizedBox(width: 60),
                Expanded(
                  child: Text('STOPS', style: AppTextStyles.tableHeader),
                ),
                Expanded(child: Text('DIST', style: AppTextStyles.tableHeader)),
                SizedBox(
                  width: 48,
                  child: Text(
                    'CAP%',
                    style: AppTextStyles.tableHeader,
                    textAlign: TextAlign.end,
                  ),
                ),
              ],
            ),
          ),
          const Divider(height: 1, color: AppColors.border),
          ...sol.routes.map(
            (route) => _RouteRow(
              route: route,
              isHovered: _hoveredRouteId == route.routeId,
              totalRoutes: sol.routes.length,
              onHoverEnter: () =>
                  setState(() => _hoveredRouteId = route.routeId),
              onHoverExit: () => setState(() => _hoveredRouteId = null),
            ),
          ),
        ],
      ),
    );
  }

  String _fmt(double v) => v.toStringAsFixed(1);
}

// ─── ROUTE CANVAS ─────────────────────────────────────────────────────────────

class _RouteCanvas extends StatelessWidget {
  final _SolutionData solution;
  final int? hoveredRouteId;
  final ValueChanged<int?> onHover;

  const _RouteCanvas({
    required this.solution,
    required this.hoveredRouteId,
    required this.onHover,
  });

  @override
  Widget build(BuildContext context) {
    if (solution.customers.isEmpty) {
      return Container(
        height: 400,
        alignment: Alignment.center,
        child: Text(
          'No route data',
          style: AppTextStyles.mono.copyWith(color: AppColors.textMuted),
        ),
      );
    }

    return AspectRatio(
      aspectRatio: 1.4,
      child: LayoutBuilder(
        builder: (context, constraints) {
          return MouseRegion(
            onExit: (_) => onHover(null),
            child: CustomPaint(
              size: Size(constraints.maxWidth, constraints.maxHeight),
              painter: _RoutePainter(
                solution: solution,
                hoveredRouteId: hoveredRouteId,
                canvasSize: Size(constraints.maxWidth, constraints.maxHeight),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _RoutePainter extends CustomPainter {
  final _SolutionData solution;
  final int? hoveredRouteId;
  final Size canvasSize;

  _RoutePainter({
    required this.solution,
    required this.hoveredRouteId,
    required this.canvasSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final padding = 24.0;
    final allNodes = [solution.depot, ...solution.customers];

    final xs = allNodes.map((n) => n.x).toList();
    final ys = allNodes.map((n) => n.y).toList();

    final minX = xs.reduce(math.min);
    final maxX = xs.reduce(math.max);
    final minY = ys.reduce(math.min);
    final maxY = ys.reduce(math.max);

    final rangeX = (maxX - minX).clamp(1.0, double.infinity);
    final rangeY = (maxY - minY).clamp(1.0, double.infinity);

    Offset toCanvas(_NodePoint n) {
      final cx = padding + (n.x - minX) / rangeX * (size.width - padding * 2);
      final cy = padding + (n.y - minY) / rangeY * (size.height - padding * 2);
      return Offset(cx, cy);
    }

    final depotPos = toCanvas(solution.depot);
    final nodePos = {for (final c in solution.customers) c.id: toCanvas(c)};

    // ── Draw routes ──────────────────────────────────────────────────────────
    for (int ri = 0; ri < solution.routes.length; ri++) {
      final route = solution.routes[ri];
      final color = AppColors.routeColors[ri % AppColors.routeColors.length];
      final isHovered = hoveredRouteId == route.routeId;
      final isDimmed = hoveredRouteId != null && !isHovered;

      final linePaint = Paint()
        ..color = isDimmed ? color.withOpacity(0.12) : color.withOpacity(0.7)
        ..strokeWidth = isHovered ? 2.0 : 1.4
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;

      final path = Path()..moveTo(depotPos.dx, depotPos.dy);
      for (final cid in route.customerIds) {
        final p = nodePos[cid];
        if (p != null) path.lineTo(p.dx, p.dy);
      }
      path.lineTo(depotPos.dx, depotPos.dy);
      canvas.drawPath(path, linePaint);
    }

    // ── Draw customer nodes ───────────────────────────────────────────────────
    for (int ri = 0; ri < solution.routes.length; ri++) {
      final route = solution.routes[ri];
      final color = AppColors.routeColors[ri % AppColors.routeColors.length];
      final isHovered = hoveredRouteId == route.routeId;
      final isDimmed = hoveredRouteId != null && !isHovered;

      final fillPaint = Paint()
        ..color = AppColors.bg2
        ..style = PaintingStyle.fill;
      final strokePaint = Paint()
        ..color = isDimmed ? color.withOpacity(0.2) : color
        ..strokeWidth = isHovered ? 2.0 : 1.5
        ..style = PaintingStyle.stroke;

      for (final cid in route.customerIds) {
        final pos = nodePos[cid];
        if (pos == null) continue;
        final r = isHovered ? 5.0 : 4.0;
        canvas.drawCircle(pos, r, fillPaint);
        canvas.drawCircle(pos, r, strokePaint);
      }
    }

    // ── Draw directional arrows and stop labels on hover ────────────────────
    if (hoveredRouteId != null) {
      final route = solution.routes.cast<_RouteData?>().firstWhere(
        (r) => r?.routeId == hoveredRouteId,
        orElse: () => null,
      );
      if (route != null) {
        final color = AppColors
            .routeColors[(route.routeId - 1) % AppColors.routeColors.length];

        // Draw arrows between consecutive stops
        final arrowPaint = Paint()
          ..color = color
          ..strokeWidth = 1.5
          ..style = PaintingStyle.stroke;

        // Helper function to draw arrow between two points
        void drawArrowBetween(Offset from, Offset to) {
          final dx = to.dx - from.dx;
          final dy = to.dy - from.dy;
          final angle = math.atan2(dy, dx);
          const circleRadius = 5.0;
          const arrowSize = 10.0;

          final lineStart = Offset(
            from.dx + circleRadius * math.cos(angle),
            from.dy + circleRadius * math.sin(angle),
          );
          final lineEnd = Offset(
            to.dx - circleRadius * math.cos(angle),
            to.dy - circleRadius * math.sin(angle),
          );

          canvas.drawLine(lineStart, lineEnd, arrowPaint);

          final midPoint = Offset(
            (lineStart.dx + lineEnd.dx) / 2,
            (lineStart.dy + lineEnd.dy) / 2,
          );

          final arrowTip = midPoint;
          final arrowLeft = Offset(
            midPoint.dx - arrowSize * math.cos(angle - math.pi / 6),
            midPoint.dy - arrowSize * math.sin(angle - math.pi / 6),
          );
          final arrowRight = Offset(
            midPoint.dx - arrowSize * math.cos(angle + math.pi / 6),
            midPoint.dy - arrowSize * math.sin(angle + math.pi / 6),
          );

          canvas.drawLine(arrowTip, arrowLeft, arrowPaint);
          canvas.drawLine(arrowTip, arrowRight, arrowPaint);
        }

        // Draw arrow from depot to first customer
        if (route.customerIds.isNotEmpty) {
          final firstCid = route.customerIds.first;
          final firstPos = nodePos[firstCid];
          if (firstPos != null) drawArrowBetween(depotPos, firstPos);
        }

        // Draw arrows between consecutive customer stops
        for (int i = 0; i < route.customerIds.length - 1; i++) {
          final fromPos = nodePos[route.customerIds[i]];
          final toPos = nodePos[route.customerIds[i + 1]];
          if (fromPos == null || toPos == null) continue;
          drawArrowBetween(fromPos, toPos);
        }

        // Draw arrow from last customer back to depot
        if (route.customerIds.isNotEmpty) {
          final lastCid = route.customerIds.last;
          final lastPos = nodePos[lastCid];
          if (lastPos != null) drawArrowBetween(lastPos, depotPos);
        }

        // Draw customer ID labels
        for (int i = 0; i < route.customerIds.length; i++) {
          final cid = route.customerIds[i];
          final pos = nodePos[cid];
          if (pos == null) continue;

          final textPainter = TextPainter(
            text: TextSpan(
              text: cid.toString(),
              style: TextStyle(
                color: color,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
            textDirection: TextDirection.ltr,
          );
          textPainter.layout();
          textPainter.paint(
            canvas,
            Offset(
              pos.dx - textPainter.width / 2,
              pos.dy - textPainter.height - 8,
            ),
          );
        }
      }
    }

    // ── Draw depot ────────────────────────────────────────────────────────────
    canvas.drawCircle(depotPos, 6, Paint()..color = AppColors.textPrimary);
    canvas.drawCircle(
      depotPos,
      6,
      Paint()
        ..color = AppColors.bg2
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(_RoutePainter old) =>
      old.hoveredRouteId != hoveredRouteId || old.solution != solution;
}

// ─── ROUTE ROW ────────────────────────────────────────────────────────────────

class _RouteRow extends StatelessWidget {
  final _RouteData route;
  final bool isHovered;
  final int totalRoutes;
  final VoidCallback onHoverEnter;
  final VoidCallback onHoverExit;

  const _RouteRow({
    required this.route,
    required this.isHovered,
    required this.totalRoutes,
    required this.onHoverEnter,
    required this.onHoverExit,
  });

  @override
  Widget build(BuildContext context) {
    final idx = route.routeId - 1;
    final color = AppColors.routeColors[idx % AppColors.routeColors.length];
    final capColor = route.capacityPct > 90
        ? AppColors.green
        : route.capacityPct > 70
        ? AppColors.amber
        : AppColors.red;

    return MouseRegion(
      onEnter: (_) => onHoverEnter(),
      onExit: (_) => onHoverExit(),
      child: AnimatedContainer(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(6),
          color: isHovered ? color.withOpacity(0.08) : Colors.transparent,
        ),
        duration: const Duration(milliseconds: 120),
        // color: isHovered ? color.withOpacity(0.08) : Colors.transparent,
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        child: Row(
          children: [
            // Route colour dot + number
            SizedBox(
              width: 60,
              child: Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: color,
                    ),
                  ),
                  const SizedBox(width: 4),
                  Text(
                    '${route.routeId}',
                    style: AppTextStyles.tableCell.copyWith(
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 3),
            // Stops
            Expanded(
              child: Text(
                '${route.numStops}',
                style: AppTextStyles.tableCell.copyWith(
                  color: AppColors.textPrimary,
                ),
              ),
            ),
            // Distance
            Expanded(
              child: Text(
                route.distance.toStringAsFixed(1),
                style: AppTextStyles.tableCellColored(color),
              ),
            ),
            // Cap%
            SizedBox(
              width: 48,
              child: Text(
                '${route.capacityPct.toStringAsFixed(0)}%',
                style: AppTextStyles.tableCellColored(capColor),
                textAlign: TextAlign.end,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── SHARED WIDGETS ──────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  const _SectionCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
      decoration: BoxDecoration(
        color: AppColors.bg2,
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Container(width: 16, height: 1, color: AppColors.textMuted),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  style: AppTextStyles.monoLabel,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              Container(width: 16, height: 1, color: AppColors.border),
            ],
          ),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }
}

class _BannerKpi extends StatelessWidget {
  final String label;
  final String value;
  final String unit;
  final Color color;
  final _SolutionData solution;
  const _BannerKpi({
    required this.label,
    required this.value,
    required this.unit,
    required this.color,
    required this.solution,
  });

  @override
  Widget build(BuildContext context) {
    final bool isSmallScreen = MediaQuery.of(context).size.width < 800;
    return Expanded(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text( //kpi labels like "VEHICLES", "TOTAL DISTANCE" etc
            label,
            style: AppTextStyles.monoLabel.copyWith(
              fontSize: 9,
              letterSpacing: 0.15,
            ),
          ),
          const SizedBox(height: 8),
          if (!isSmallScreen) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Flexible(
                  child: RichText(
                    textAlign: TextAlign.center,
                    text: TextSpan(
                      children: [
                        TextSpan(
                          text: value,
                          style: GoogleFonts.syne(
                            fontSize: 28,
                            fontWeight: FontWeight.w800,
                            color: solution.customers.isEmpty
                                ? AppColors.textMuted
                                : color,
                            height: 1.0,
                          ),
                        ),
                        if (unit.isNotEmpty)
                          TextSpan(
                            text: ' $unit',
                            style: GoogleFonts.syne(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: solution.customers.isEmpty
                                  ? AppColors.textMuted
                                  : color.withOpacity(0.7),
                            ),
                          ),
                      ],
                    ),
                  ),
                )
              ],
            ),
          ] else ...[
            Text(
              value,
              style: GoogleFonts.syne(
                fontSize: 9,
                fontWeight: FontWeight.w800,
                color: color,
                height: 1.0,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(width: 6),
            if (unit.isNotEmpty)
              Text(
                unit,
                style: GoogleFonts.syne(
                  fontSize: 9,
                  fontWeight: FontWeight.w600,
                  color: color.withOpacity(0.7),
                ),
              ),
          ],
        ],
      ),
    );
  }
}

class _BannerDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
    width: 1,
    height: 60,
    color: AppColors.textPrimary.withOpacity(0.2),
    margin: const EdgeInsets.symmetric(horizontal: 4),
  );
}

class _ExportButton extends StatefulWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;
  final bool disabled;
  const _ExportButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
    this.disabled = false,
  });

  @override
  State<_ExportButton> createState() => _ExportButtonState();
}

class _ExportButtonState extends State<_ExportButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isDisabled = widget.disabled;
    final displayColor = isDisabled ? AppColors.textMuted : widget.color;

    return MouseRegion(
      cursor: isDisabled ? SystemMouseCursors.basic : SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: isDisabled ? null : widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 11, horizontal: 14),
          decoration: BoxDecoration(
            color: (_hovered && !isDisabled)
                ? displayColor.withOpacity(0.14)
                : displayColor.withOpacity(0.06),
            border: Border.all(
              color: (_hovered && !isDisabled)
                  ? displayColor.withOpacity(0.55)
                  : displayColor.withOpacity(0.25),
            ),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            children: [
              Icon(widget.icon, size: 16, color: displayColor),
              const SizedBox(width: 8),
              Text(
                widget.label,
                style: GoogleFonts.jetBrainsMono(
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                  color: displayColor,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
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
