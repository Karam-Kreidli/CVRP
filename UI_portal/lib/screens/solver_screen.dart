import 'dart:async';
import 'package:RouteIQ_UI/services/benchmark_service.dart';
import 'package:RouteIQ_UI/services/solver_controller.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/widgets/Tag.dart';
import 'package:RouteIQ_UI/widgets/sectionlabel_line.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ─── SOLVER CONSOLE SCREEN ───────────────────────────────────────────────────
//
// Pure UI layer over SolverController.instance.
// Owns NO solver state — all of that lives in the singleton controller.
//
// Behaviour on mount:
//   • status == complete && solutionViewed → auto-reset so user can upload fresh file
//   • pendingNavigation == true  → show countdown overlay toast, navigate when done
//   • Otherwise render current state normally.

class SolverScreen extends StatefulWidget {
  const SolverScreen({super.key});

  @override
  State<SolverScreen> createState() => _SolverScreenState();
}

class _SolverScreenState extends State<SolverScreen> {
  // ── Local UI-only state ────────────────────────────────────────────────────
  bool _isDragOver = false;

  /// Tracks the previous status so we can detect transitions in the listener.
  SolverStatus? _lastKnownStatus;

  // ── Countdown overlay (exact match of old solver screen) ──────────────────
  OverlayEntry? _countdownOverlay;
  Timer? _countdownTimer;

  SolverController get _ctrl => SolverController.instance;

  @override
  void initState() {
    super.initState();
    _lastKnownStatus = _ctrl.status;
    _ctrl.addListener(_onControllerChanged);

    // ── Auto-reset: if returning after having already viewed the solution ────
    if (_ctrl.status == SolverStatus.complete &&
        !_ctrl.pendingNavigation &&
        _ctrl.solutionViewed) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _ctrl.resetFile();
      });
      return;
    }

    // ── Pending navigation: solve completed while user was on another page ───
    if (_ctrl.pendingNavigation) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        if (!_ctrl.pendingNavigation || _countdownOverlay != null) return;
        final jid = _ctrl.jobId;
        if (jid != null) {
          _ctrl.clearPendingNavigation();
          _showCountdownToast(jid);
        }
      });
    }
  }

  @override
  void dispose() {
    _ctrl.removeListener(_onControllerChanged);
    _dismissCountdownToast();
    super.dispose();
  }

  // ── Controller listener ───────────────────────────────────────────────────

  void _onControllerChanged() {
    if (!mounted) return;

    final prev = _lastKnownStatus;
    _lastKnownStatus = _ctrl.status;

    setState(() {});

    if (prev != SolverStatus.error && _ctrl.status == SolverStatus.error) {
      _showErrorSnack(
        _ctrl.lastErrorMessage ??
            'Could not reach solver — check your connection',
      );
    }

    if (_ctrl.pendingNavigation && _countdownOverlay == null) {
      final jid = _ctrl.jobId;
      if (jid != null) {
        _ctrl.clearPendingNavigation();
        _showCountdownToast(jid);
      }
    }
  }

  // ── File picking ──────────────────────────────────────────────────────────

  Future<void> _pickFile() async {
    if (_ctrl.status == SolverStatus.running) return;
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['vrp'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.bytes == null) return;
    _ctrl.loadFile(
      name: file.name,
      bytes: file.bytes!,
      sizeKb: ((file.size ?? 0) / 1024).toStringAsFixed(1),
    );
    BenchmarkService.instance.setFile(file.bytes!, file.name);
  }

  // ── Error snackbar ────────────────────────────────────────────────────────

  void _showErrorSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          message,
          style: AppTextStyles.subheading.copyWith(
            color: AppColors.textPrimary,
          ),
        ),
        backgroundColor: AppColors.red,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(6),
          side: const BorderSide(color: AppColors.red),
        ),
      ),
    );
  }

  // ── Countdown overlay toast ────────────────────────────────────────────────

  int _countdownRemaining = 0;

  void _showCountdownToast(String jobId) {
    _dismissCountdownToast();
    _ctrl.lockSolutionNav();
    _countdownRemaining = kNavigationDelaySeconds;

    late OverlayEntry entry;
    entry = OverlayEntry(
      builder: (context) => Positioned(
        top: 28,
        left: 16,
        right: 16,
        child: Center(
          child: Material(
            color: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              decoration: BoxDecoration(
                color: AppColors.bg2,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: AppColors.green.withOpacity(0.5),
                  width: 1.5,
                ),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.green.withOpacity(0.12),
                    blurRadius: 20,
                    spreadRadius: 2,
                  ),
                  BoxShadow(
                    color: Colors.black.withOpacity(0.4),
                    blurRadius: 16,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Green check icon
                  Container(
                    width: 28,
                    height: 28,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.green.withOpacity(0.12),
                      border: Border.all(
                        color: AppColors.green.withOpacity(0.4),
                        width: 1.5,
                      ),
                    ),
                    child: const Icon(
                      Icons.check_rounded,
                      size: 14,
                      color: AppColors.green,
                    ),
                  ),
                  const SizedBox(width: 12),

                  // Label
                  Flexible(
                    child: RichText(
                      text: TextSpan(
                        children: [
                          TextSpan(
                            text: 'SOLVE COMPLETE',
                            style: GoogleFonts.syne(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: AppColors.green,
                              letterSpacing: 0.5,
                            ),
                          ),
                          TextSpan(
                            text: '  —  showing solution in',
                            style: GoogleFonts.jetBrainsMono(
                              fontSize: 11,
                              fontWeight: FontWeight.w400,
                              color: AppColors.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),

                  // Countdown circle
                  Container(
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.bg3,
                      border: Border.all(
                        color: AppColors.green.withOpacity(0.5),
                        width: 1.5,
                      ),
                    ),
                    child: Center(
                      child: Text(
                        '$_countdownRemaining',
                        style: GoogleFonts.syne(
                          fontSize: 13,
                          fontWeight: FontWeight.w800,
                          color: AppColors.green,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );

    _countdownOverlay = entry;
    Overlay.of(context).insert(entry);

    // Start timer immediately (not inside builder) to ensure it runs
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) {
        _dismissCountdownToast();
        return;
      }
      _countdownRemaining--;
      if (_countdownRemaining <= 0) {
        _dismissCountdownToast();
        _ctrl.markSolutionViewed();
        ShellNav.pushSolution(jobId: jobId);
        return;
      }
      // Trigger overlay rebuild to show updated countdown
      _countdownOverlay?.markNeedsBuild();
    });
  }

  void _dismissCountdownToast() {
    _countdownTimer?.cancel();
    _countdownTimer = null;
    _countdownOverlay?.remove();
    _countdownOverlay = null;
    _ctrl.unlockSolutionNav();
  }

  // ── Build ─────────────────────────────────────────────────────────────────

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
                _buildMainLayout(context),
              ],
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
            Text('Solver Console', style: AppTextStyles.displayMedium),
            if (_ctrl.status == SolverStatus.running) ...[
              const SizedBox(width: 12),
              Tag(label: 'RUNNING', color: AppColors.cyan, fixWidth: false),
            ],
            if (_ctrl.status == SolverStatus.complete) ...[
              const SizedBox(width: 12),
              Tag(label: 'COMPLETE', color: AppColors.green, fixWidth: false),
            ],
          ],
        ),
        const SizedBox(height: 6),
        // ── CHANGE: updated pipeline stage count from 5 to 4 ─────────────────
        Text(
          'Upload a .vrp instance and run the full 4-stage pipeline',
          style: AppTextStyles.mono,
        ),
      ],
    );
  }

  Widget _buildMainLayout(BuildContext context) {
    final narrow = MediaQuery.of(context).size.width < 1050;
    if (narrow) {
      return Column(
        children: [
          _buildLeftPanel(),
          const SizedBox(height: 14),
          _buildRightPanel(),
        ],
      );
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(width: 380, child: _buildLeftPanel()),
        const SizedBox(width: 14),
        Expanded(child: _buildRightPanel()),
      ],
    );
  }

  // ── Left panel ────────────────────────────────────────────────────────────

  Widget _buildLeftPanel() {
    return Column(
      children: [
        _buildInstanceFileCard(),
        const SizedBox(height: 14),
        // _buildConfigCard(),
        const SizedBox(height: 14),
        _buildRunButton(),
      ],
    );
  }

  Widget _buildInstanceFileCard() {
    final info = _ctrl.instanceInfo;
    final isRunning = _ctrl.status == SolverStatus.running;

    return DropTarget(
      onDragEntered: (_) => setState(() => _isDragOver = true),
      onDragExited: (_) => setState(() => _isDragOver = false),
      onDragDone: (details) async {
        if (isRunning) return;
        setState(() => _isDragOver = false);
        if (details.files.isEmpty) return;
        final file = details.files.first;
        if (!file.name.toLowerCase().endsWith('.vrp')) return;
        final bytes = await file.readAsBytes();
        final sizeKb = (bytes.length / 1024).toStringAsFixed(1);
        _ctrl.loadFile(name: file.name, bytes: bytes, sizeKb: sizeKb);
        BenchmarkService.instance.setFile(bytes, file.name);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          boxShadow: _isDragOver
              ? [
                  BoxShadow(
                    color: AppColors.cyan.withOpacity(0.3),
                    blurRadius: 12,
                    spreadRadius: 2,
                  ),
                ]
              : null,
        ),
        child: _SectionCard(
          title: 'INSTANCE FILE',
          dashed: true,
          dashedActive: info != null || _isDragOver,
          highlightBorder: _isDragOver,
          child: info != null ? _buildFileLoaded(info) : _buildFileDropZone(),
        ),
      ),
    );
  }

  Widget _buildFileDropZone() {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: _pickFile,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Icon(
                Icons.upload_file_outlined,
                size: 32,
                color: _isDragOver ? AppColors.cyan : AppColors.textMuted,
              ),
              const SizedBox(height: 10),
              Text(
                'Drop .vrp file here',
                style: AppTextStyles.subheading.copyWith(
                  color: _isDragOver ? AppColors.cyan : AppColors.textSecondary,
                ),
              ),
              const SizedBox(height: 4),
              Text('or click to browse', style: AppTextStyles.monoSmall),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFileLoaded(InstanceInfo info) {
    final isRunning = _ctrl.status == SolverStatus.running;
    final isComplete = _ctrl.status == SolverStatus.complete;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // File name row
        Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: AppColors.cyanFaint,
                border: Border.all(color: AppColors.cyanDim),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.description_outlined,
                color: AppColors.cyan,
                size: 18,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    info.fileName,
                    style: AppTextStyles.mono.copyWith(
                      color: AppColors.textPrimary,
                      fontWeight: FontWeight.w500,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text(
                    '${info.fileSizeKb} KB · CVRP instance',
                    style: AppTextStyles.monoSmall,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Tag(label: 'LOADED', color: AppColors.green, fixWidth: false),
          ],
        ),
        const SizedBox(height: 14),
        GridView.count(
          crossAxisCount: 2,
          crossAxisSpacing: 8,
          mainAxisSpacing: 8,
          childAspectRatio: 2.6,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            _InfoTile(label: 'Customers', value: info.numCustomers.toString()),
            _InfoTile(label: 'Depot', value: 'Node ${info.depotNode}'),
            _InfoTile(
              label: 'Capacity',
              value: '${_fmtInt(info.capacity)} units',
            ),
            _InfoTile(
              label: 'Demand Total',
              value: '${_fmtInt(info.totalDemand)} units',
            ),
          ],
        ),
        const SizedBox(height: 12),
        // Delete file button — disabled while running
        MouseRegion(
          cursor: isRunning
              ? SystemMouseCursors.forbidden
              : SystemMouseCursors.click,
          child: GestureDetector(
            onTap: isRunning ? null : _ctrl.resetFile,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.bg3,
                border: Border.all(
                  color: (isRunning || isComplete)
                      ? AppColors.textMuted
                      : AppColors.red,
                ),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.delete_outline,
                    size: 16,
                    color: (isRunning || isComplete)
                        ? AppColors.textMuted
                        : AppColors.red,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Delete File',
                    style: AppTextStyles.monoSmall.copyWith(
                      color: (isRunning || isComplete)
                          ? AppColors.textMuted
                          : AppColors.red,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRunButton() {
    final isRunning = _ctrl.status == SolverStatus.running;
    final isComplete = _ctrl.status == SolverStatus.complete;
    final noFile = _ctrl.instanceInfo == null;
    return _HoverButton(
      label: isComplete
          ? 'SOLVE COMPLETE'
          : isRunning
          ? 'STOP'
          : 'RUN SOLVER',
      color: isComplete
          ? AppColors.green
          : isRunning
          ? AppColors.red
          : AppColors.cyan,
      icon: isComplete
          ? Icons.check_rounded
          : isRunning
          ? Icons.stop
          : Icons.play_arrow,
      onTap: noFile
          ? null
          : (isComplete
                ? null
                : isRunning
                ? _ctrl.stopSolver
                : _ctrl.runSolver),
    );
  }

  // ── Right panel ───────────────────────────────────────────────────────────

  Widget _buildRightPanel() {
    return Column(
      children: [
        _buildPipelineMonitor(),
        const SizedBox(height: 14),
        _buildLiveMetrics(),
        const SizedBox(height: 14),
        _buildConsoleLog(),
        // ── Benchmark results — visible after solve completes ─────────────
        if (_ctrl.status == SolverStatus.complete || _ctrl.solutionViewed) ...[
          const SizedBox(height: 14),
          _buildBenchmarkCard(),
        ],
      ],
    );
  }

  // Pipeline Monitor
  Widget _buildPipelineMonitor() {
    final m = _ctrl.metrics;
    final iterPct = m.maxIterations > 0 ? m.iteration / m.maxIterations : 0.0;

    // ── CHANGE: episode step progress bar (new for 4-stage architecture) ──────
    //
    // The RL agent runs 50 steps per episode (episodeStep tracks 0→50).
    // Each step invokes HGS with 500–1500 iterations (FREE/LOCK/PUSH/FORCE).
    // We show both the episode step progress and cumulative HGS iterations.
    final stepPct = m.episodeStepMax > 0
        ? m.episodeStep / m.episodeStepMax
        : 0.0;

    return _SectionCard(
      title: 'PIPELINE MONITOR',
      child: Column(
        children: [
          ..._ctrl.stages.map((s) => _PipelineStageRow(stage: s)),
          const SizedBox(height: 16),

          // ── CHANGE: RL episode step progress ─────────────────────────────
          if (m.episodeStepMax > 0) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('RL EPISODE STEP', style: AppTextStyles.monoLabel),
                Row(
                  children: [
                    if (m.currentAction.isNotEmpty)
                      Container(
                        margin: const EdgeInsets.only(right: 8),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: AppColors.amber.withOpacity(0.12),
                          border: Border.all(
                            color: AppColors.amber.withOpacity(0.4),
                          ),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          m.currentAction,
                          style: AppTextStyles.monoSmall.copyWith(
                            color: AppColors.amber,
                            fontSize: 9,
                          ),
                        ),
                      ),
                    Text(
                      '${m.episodeStep} / ${m.episodeStepMax}',
                      style: AppTextStyles.monoSmall.copyWith(
                        color: AppColors.amber,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: stepPct.clamp(0.0, 1.0),
                minHeight: 4,
                backgroundColor: AppColors.border,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  AppColors.amber,
                ),
              ),
            ),
            const SizedBox(height: 12),
          ],

          // HGS cumulative iterations progress bar
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('HGS ITERATIONS', style: AppTextStyles.monoLabel),
              Text(
                '${_fmtInt(m.iteration)} / ${_fmtInt(m.maxIterations)}',
                style: AppTextStyles.monoSmall.copyWith(color: AppColors.cyan),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: iterPct.clamp(0.0, 1.0),
              minHeight: 4,
              backgroundColor: AppColors.border,
              valueColor: const AlwaysStoppedAnimation<Color>(AppColors.cyan),
            ),
          ),
        ],
      ),
    );
  }

  // Live Metrics
  Widget _buildLiveMetrics() {
    final m = _ctrl.metrics;
    return _SectionCard(
      title: 'LIVE METRICS',
      child: GridView.count(
        crossAxisCount: 2,
        crossAxisSpacing: 10,
        mainAxisSpacing: 10,
        childAspectRatio: 2.0,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        children: [
          _MetricTile(
            label: 'Current NV',
            value: m.currentNv > 0 ? m.currentNv.toString() : '—',
            color: AppColors.cyan,
          ),
          _MetricTile(
            label: 'Best NV',
            value: m.bestNv > 0 ? m.bestNv.toString() : '—',
            color: AppColors.green,
          ),
          _MetricTile(
            label: 'Current TD',
            value: m.currentTd > 0 ? m.currentTd.toStringAsFixed(1) : '—',
            color: AppColors.amber,
          ),
          _MetricTile(
            label: 'Best TD',
            value: m.bestTd > 0 ? m.bestTd.toStringAsFixed(1) : '—',
            color: AppColors.green,
          ),
          _MetricTile(
            label: 'Current Score',
            value: m.currentScore > 0 ? _fmtInt(m.currentScore.round()) : '—',
            color: AppColors.textPrimary,
          ),
          _MetricTile(
            label: 'Best Score',
            value: m.bestScore > 0 ? _fmtInt(m.bestScore.round()) : '—',
            color: AppColors.green,
          ),
        ],
      ),
    );
  }

  // Console Log
  Widget _buildConsoleLog() {
    final lines = _ctrl.metrics.logLines;
    if (lines.isEmpty) return const SizedBox.shrink();
    return _SectionCard(
      title: 'CONSOLE LOG',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: lines.map((line) {
          // ── CHANGE: updated log tag colors for 4-stage architecture ──────────
          //
          // Removed:  [GNN] (GNN Observer no longer exists)
          // Added:    [FE]  (Feature Extractor, stage 1) → green
          //           [PPO] (PPO Trainer, stage 4)       → purple
          // Kept:     [FM]  (Fleet Manager) → cyan
          //           [HGS] (HGS Engine)   → amber
          Color c = AppColors.textMuted;
          if (line.contains('[FE]')) c = AppColors.green;
          if (line.contains('[FM]')) c = AppColors.cyan;
          if (line.contains('[HGS]')) c = AppColors.amber;
          if (line.contains('[PPO]')) c = AppColors.purple;
          if (line.contains('New best')) c = AppColors.amber;
          return Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Text(
              line,
              style: AppTextStyles.monoSmall.copyWith(
                color: c,
                fontSize: 10,
                height: 1.7,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  // Benchmark Results — visible after solve completes
  Widget _buildBenchmarkCard() {
    final m = _ctrl.metrics;
    return _SectionCard(
      title: 'BENCHMARK RESULTS',
      child: GridView.count(
        crossAxisCount: 2,
        crossAxisSpacing: 10,
        mainAxisSpacing: 10,
        childAspectRatio: 2.0,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        children: [
          _MetricTile(
            label: 'Best Vehicles',
            value: m.bestNv > 0 ? m.bestNv.toString() : '—',
            color: AppColors.cyan,
          ),
          _MetricTile(
            label: 'Best Distance',
            value: m.bestTd > 0 ? m.bestTd.toStringAsFixed(1) : '—',
            color: AppColors.amber,
          ),
          _MetricTile(
            label: 'Best Score',
            value: m.bestScore > 0 ? _fmtInt(m.bestScore.round()) : '—',
            color: AppColors.green,
          ),
          _MetricTile(
            label: 'HGS Iterations',
            value: m.iteration > 0 ? _fmtInt(m.iteration) : '—',
            color: AppColors.textSecondary,
          ),
        ],
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  String _fmtInt(int v) {
    final s = v.toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }
}

// ─── PIPELINE STAGE ROW ───────────────────────────────────────────────────────

class _PipelineStageRow extends StatefulWidget {
  final PipelineStageData stage;
  const _PipelineStageRow({required this.stage});

  @override
  State<_PipelineStageRow> createState() => _PipelineStageRowState();
}

class _PipelineStageRowState extends State<_PipelineStageRow>
    with SingleTickerProviderStateMixin {
  late final AnimationController _rotCtrl;

  @override
  void initState() {
    super.initState();
    _rotCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _rotCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.stage;
    final Color color;
    final String label;

    switch (s.status) {
      case StageStatus.done:
        color = AppColors.green;
        label = 'DONE';
        break;
      case StageStatus.running:
        color = AppColors.cyan;
        label = 'RUNNING';
        break;
      case StageStatus.error:
        color = AppColors.red;
        label = 'ERROR';
        break;
      case StageStatus.waiting:
        color = AppColors.textMuted;
        label = 'WAITING';
        break;
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 150),
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
      decoration: BoxDecoration(
        color: s.status == StageStatus.running
            ? AppColors.cyanFaint
            : Colors.transparent,
        borderRadius: BorderRadius.circular(6),
        border: Border(
          left: BorderSide(
            color: s.status == StageStatus.running
                ? AppColors.cyan
                : Colors.transparent,
            width: 2,
          ),
        ),
      ),
      child: Row(
        children: [
          // Stage icon
          SizedBox(
            width: 28,
            height: 28,
            child: s.status == StageStatus.running
                ? AnimatedBuilder(
                    animation: _rotCtrl,
                    builder: (_, __) => Transform.rotate(
                      angle: _rotCtrl.value * 6.283,
                      child: _StageIcon(color: color, status: s.status),
                    ),
                  )
                : _StageIcon(color: color, status: s.status),
          ),
          const SizedBox(width: 12),

          // Name + model
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  s.name,
                  style: AppTextStyles.subheading.copyWith(
                    color: s.status == StageStatus.waiting
                        ? AppColors.textMuted
                        : AppColors.textPrimary,
                    fontSize: 12,
                  ),
                ),
                Text(
                  s.model,
                  style: AppTextStyles.monoSmall.copyWith(fontSize: 9),
                ),
              ],
            ),
          ),

          // Status tag + time
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Tag(label: label, color: color, fixWidth: false),
              const SizedBox(height: 3),
              Text(
                s.elapsedSeconds != null
                    ? '${s.elapsedSeconds!.toStringAsFixed(1)}s'
                    : '—',
                style: AppTextStyles.monoSmall.copyWith(fontSize: 9),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StageIcon extends StatelessWidget {
  final Color color;
  final StageStatus status;
  const _StageIcon({required this.color, required this.status});

  @override
  Widget build(BuildContext context) {
    final icon = status == StageStatus.done
        ? Icons.check
        : status == StageStatus.error
        ? Icons.close
        : Icons.radio_button_unchecked;
    return Container(
      width: 28,
      height: 28,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(color: color, width: 1.5),
        color: color.withOpacity(0.08),
      ),
      child: Icon(icon, size: 14, color: color),
    );
  }
}

// ─── SHARED WIDGETS ──────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  final bool dashed;
  final bool dashedActive;
  final bool highlightBorder;

  const _SectionCard({
    required this.title,
    required this.child,
    this.dashed = false,
    this.dashedActive = false,
    this.highlightBorder = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
      decoration: BoxDecoration(
        color: highlightBorder ? AppColors.cyanFaint : AppColors.bg2,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: highlightBorder
              ? AppColors.cyan
              : (dashedActive ? AppColors.cyanDim : AppColors.border),
          width: highlightBorder ? 2.0 : 1.5,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          SectionLabel(title: title),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }
}

class _InfoTile extends StatelessWidget {
  final String label;
  final String value;
  const _InfoTile({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.bg3,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: AppTextStyles.monoLabel.copyWith(fontSize: 9)),
          const SizedBox(height: 3),
          Text(
            value,
            style: AppTextStyles.monoSmall.copyWith(
              color: AppColors.cyan,
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _MetricTile({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.bg3,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: AppTextStyles.monoLabel.copyWith(fontSize: 9)),
          const SizedBox(height: 4),
          Text(
            value,
            style: GoogleFonts.syne(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: color,
              height: 1.0,
            ),
          ),
        ],
      ),
    );
  }
}

class _HoverButton extends StatefulWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback? onTap;

  const _HoverButton({
    required this.label,
    required this.color,
    required this.icon,
    this.onTap,
  });

  @override
  State<_HoverButton> createState() => _HoverButtonState();
}

class _HoverButtonState extends State<_HoverButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final disabled = widget.onTap == null;
    return MouseRegion(
      cursor: disabled
          ? SystemMouseCursors.forbidden
          : SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: (_hovered && !disabled)
                ? widget.color.withOpacity(0.18)
                : widget.color.withOpacity(0.08),
            border: Border.all(
              color: disabled
                  ? AppColors.border
                  : widget.color.withOpacity(0.5),
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  widget.icon,
                  size: 16,
                  color: disabled ? AppColors.textMuted : widget.color,
                ),
                const SizedBox(width: 8),
                Text(
                  widget.label,
                  style: GoogleFonts.syne(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.08,
                    color: disabled ? AppColors.textMuted : widget.color,
                  ),
                ),
              ],
            ),
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
