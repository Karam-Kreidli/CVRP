import 'dart:async';
import 'dart:convert';
import 'package:RouteIQ_UI/services/benchmark_service.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:http/http.dart' as http;

// ─── CONSTANTS ────────────────────────────────────────────────────────────────

const _kBaseUrl = ApiConfig.baseUrl;

/// Seconds to wait on the Solver page after completion before auto-navigating
/// to the Solution page.
const kNavigationDelaySeconds = 15;

// ─── ENUMS & MODELS ──────────────────────────────────────────────────────────

enum StageStatus { waiting, running, done, error }

enum SolverStatus { idle, running, complete, error, stopped }

class InstanceInfo {
  final String fileName;
  final String fileSizeKb;
  final int numCustomers;
  final int depotNode;
  final int capacity;
  final int totalDemand;
  final int nvMin;
  final List<int> fileBytes;

  const InstanceInfo({
    required this.fileName,
    required this.fileSizeKb,
    required this.numCustomers,
    required this.depotNode,
    required this.capacity,
    required this.totalDemand,
    required this.nvMin,
    required this.fileBytes,
  });
}

class PipelineStageData {
  final int stageNum;
  final String name;
  final String model;
  StageStatus status;
  double? elapsedSeconds;

  PipelineStageData({
    required this.stageNum,
    required this.name,
    required this.model,
    this.status = StageStatus.waiting,
    this.elapsedSeconds,
  });
}

// ─── CHANGE: LiveMetrics extended with RL-specific episode fields ─────────────
//
// Added:
//   • currentAction   — the Fleet Manager's last action name (FREE_SAME, PUSH_NEW, etc.)
//   • episodeStep     — current step index within the 50-step episode (0–50)
//   • episodeStepMax  — total steps per episode (always 50, exposed for progress math)
//
// These are surfaced in the Pipeline Monitor so the user can see which HGS
// strategy the RL agent just chose and how far through the episode we are.

class LiveMetrics {
  final int currentNv;
  final int bestNv;
  final double currentTd;
  final double bestTd;
  final double currentScore;
  final double bestScore;
  final int iteration;
  final int maxIterations;
  final List<String> logLines;

  // ── NEW fields ───────────────────────────────────────────────────────────
  /// Last action chosen by the Fleet Manager (e.g. "PUSH_NEW", "LOCK_SAME").
  /// Empty string before the first step completes.
  final String currentAction;

  /// Current episode step index (0 = initial solve, 1–50 = RL steps).
  final int episodeStep;

  /// Fixed upper bound for episodeStep (always 50 per episode).
  final int episodeStepMax;

  const LiveMetrics({
    this.currentNv = 0,
    this.bestNv = 0,
    this.currentTd = 0,
    this.bestTd = 0,
    this.currentScore = 0,
    this.bestScore = 0,
    this.iteration = 0,
    this.maxIterations = 25000,
    this.logLines = const [],
    // NEW
    this.currentAction = '',
    this.episodeStep = 0,
    this.episodeStepMax = 50,
  });
}

// ─── SOLVER CONTROLLER ────────────────────────────────────────────────────────
//
// Singleton ChangeNotifier that owns ALL solver state and the poll timer.
// Lives at app level — survives navigation.
//
// Lifecycle:
//   AppShell listens → shows toast when complete + user is away from solver page
//   SolverScreen listens → renders current state, drives navigation when on page
//
// The timer runs here regardless of which page is visible. When the solve
// completes while the user is away, `pendingNavigation` is set to true.
// SolverScreen checks this on mount and starts the countdown then navigates.

class SolverController extends ChangeNotifier {
  // ── Singleton ─────────────────────────────────────────────────────────────
  static final SolverController instance = SolverController._();
  SolverController._();

  // ── State ─────────────────────────────────────────────────────────────────
  InstanceInfo? instanceInfo;
  SolverStatus status = SolverStatus.idle;
  String? jobId;
  LiveMetrics metrics = const LiveMetrics();

  /// Cached solution data (raw JSON) so SolutionScreen can restore it when
  /// the user navigates away and comes back. Cleared when a new solve starts.
  Map<String, dynamic>? cachedSolutionJson;
  String? cachedSolutionJobId;

  /// true = countdown toast is showing, disable Solution nav in sidebar.
  /// Prevents user from manually navigating during auto-redirect countdown.
  bool solutionNavLocked = false;

  /// true = solve completed while the user was away; SolverScreen shows the
  /// countdown toast on mount and navigates to Solution.
  bool pendingNavigation = false;

  /// true = user has already been taken to the Solution screen for this solve.
  /// SolverScreen uses this to auto-reset when the user returns.
  bool solutionViewed = false;

  /// Last error message from the solver, shown in the error snackbar.
  String? lastErrorMessage;

  // ─── CHANGE: 4 stages replacing the old 5-stage pipeline ─────────────────
  //
  // Removed:  Stage 1 GNN Observer, Stage 4 Route Driver, Stage 5 MACA Trainer
  // Added:    Stage 1 Feature Extractor, Stage 4 PPO Trainer
  //
  // Stage keys ("1"–"4") must match what the backend returns in
  // stage_statuses and stage_times_seconds inside GET /status/{job_id}.

  final List<PipelineStageData> stages = [
    PipelineStageData(
      stageNum: 1,
      name: 'Feature Extractor',
      model: 'Hand-crafted (7-dim)',
    ),
    PipelineStageData(
      stageNum: 2,
      name: 'Fleet Manager',
      model: 'Actor-Critic PPO',
    ),
    PipelineStageData(
      stageNum: 3,
      name: 'HGS Engine',
      model: 'Hybrid Genetic Search',
    ),
    PipelineStageData(
      stageNum: 4,
      name: 'PPO Trainer',
      model: 'Reward Propagation',
    ),
  ];

  Timer? _pollTimer;

  // ── File loading ──────────────────────────────────────────────────────────

  void loadFile({
    required String name,
    required List<int> bytes,
    required String sizeKb,
  }) {
    int customers = 0;
    int capacity = 0;
    int totalDemand = 0;

    final content = String.fromCharCodes(bytes);
    for (final line in content.split('\n')) {
      final t = line.trim();
      if (t.startsWith('DIMENSION')) {
        final parts = t.split(':');
        if (parts.length > 1)
          customers = (int.tryParse(parts[1].trim()) ?? 1) - 1;
      }
      if (t.startsWith('CAPACITY')) {
        final parts = t.split(':');
        if (parts.length > 1) capacity = int.tryParse(parts[1].trim()) ?? 0;
      }
    }
    totalDemand = customers * 94; // rough estimate — backend overrides

    instanceInfo = InstanceInfo(
      fileName: name,
      fileSizeKb: sizeKb,
      numCustomers: customers,
      depotNode: 1,
      capacity: capacity,
      totalDemand: totalDemand,
      nvMin: capacity > 0 ? (totalDemand / capacity).ceil() : 0,
      fileBytes: bytes,
    );
    status = SolverStatus.idle;
    pendingNavigation = false;
    solutionViewed = false;
    lastErrorMessage = null;
    _resetStages();
    metrics = const LiveMetrics();
    notifyListeners();
  }

  void resetFile() {
    instanceInfo = null;
    status = SolverStatus.idle;
    pendingNavigation = false;
    solutionViewed = false;
    lastErrorMessage = null;
    jobId = null;
    metrics = const LiveMetrics();
    _resetStages();
    _pollTimer?.cancel();
    notifyListeners();
  }

  void _resetStages() {
    for (final s in stages) {
      s.status = StageStatus.waiting;
      s.elapsedSeconds = null;
    }
  }

  // ── Solver control ────────────────────────────────────────────────────────

  Future<void> runSolver() async {
    if (instanceInfo == null) return;
    if (status == SolverStatus.running) return;

    status = SolverStatus.running;
    pendingNavigation = false;
    solutionViewed = false;
    lastErrorMessage = null;
    // Clear cached solution when starting a new solve
    cachedSolutionJson = null;
    cachedSolutionJobId = null;
    // Reset benchmark results when starting a new solve
    BenchmarkService.instance.resetResults();
    _resetStages();
    metrics = const LiveMetrics();
    notifyListeners();

    final info = instanceInfo!;

    // POST /solve — upload the file
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_kBaseUrl/solve'),
      );
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          info.fileBytes,
          filename: info.fileName,
        ),
      );
      request.fields['track'] = 'cvrp';
      request.fields['mode'] = 'competition';
      request.fields['time_limit_seconds'] = '300';

      final response = await request.send().timeout(
        const Duration(seconds: 15),
      );
      if (response.statusCode != 202) {
        lastErrorMessage = 'Solver returned ${response.statusCode}';
        status = SolverStatus.error;
        notifyListeners();
        return;
      }

      final body =
          jsonDecode(await response.stream.bytesToString())
              as Map<String, dynamic>;
      jobId = body['job_id'] as String;

      // Update instance info with accurate server-parsed values
      instanceInfo = InstanceInfo(
        fileName: body['instance_name'] as String,
        fileSizeKb: info.fileSizeKb,
        numCustomers: (body['num_nodes'] as int) - 1,
        depotNode: 1,
        capacity: body['vehicle_capacity'] as int,
        totalDemand: body['total_demand'] as int,
        nvMin: body['nv_min'] as int,
        fileBytes: info.fileBytes,
      );
      notifyListeners();
    } catch (e) {
      lastErrorMessage = 'Could not reach solver — check your connection';
      status = SolverStatus.error;
      notifyListeners();
      return;
    }

    // Start polling — timer lives here, NOT in the screen widget
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(
      const Duration(milliseconds: 500),
      (_) => _pollStatus(),
    );
  }

  Future<void> stopSolver() async {
    _pollTimer?.cancel();
    if (jobId != null) {
      try {
        await http.post(Uri.parse('$_kBaseUrl/stop/$jobId'));
      } catch (_) {}
    }
    status = SolverStatus.stopped;
    notifyListeners();
  }

  // ── Polling ───────────────────────────────────────────────────────────────

  Future<void> _pollStatus() async {
    if (jobId == null) return;
    try {
      final res = await http.get(Uri.parse('$_kBaseUrl/status/$jobId'));
      if (res.statusCode != 200) return;
      final body = jsonDecode(res.body) as Map<String, dynamic>;

      final st = body['status'] as String;
      final stageStatuses = body['stage_statuses'] as Map<String, dynamic>;
      final stageTimes = body['stage_times_seconds'] as Map<String, dynamic>;
      final logLines = (body['log_lines'] as List).cast<String>();

      // ── CHANGE: iterate over 4 stages (keys "1"–"4") ─────────────────────
      for (final s in stages) {
        final key = s.stageNum.toString();
        final rawStatus = stageStatuses[key] as String? ?? 'waiting';
        s.status = _parseStage(rawStatus);
        s.elapsedSeconds = (stageTimes[key] as num?)?.toDouble();
      }

      // ── CHANGE: parse new RL episode fields from the status response ──────
      //
      // current_action  — last Fleet Manager action (e.g. "PUSH_NEW")
      // episode_step    — step index within the 50-step RL episode
      // episode_step_max — always 50, passed through for convenience

      metrics = LiveMetrics(
        currentNv: (body['current_nv'] as num?)?.toInt() ?? metrics.currentNv,
        bestNv: (body['best_nv'] as num?)?.toInt() ?? metrics.bestNv,
        currentTd:
            (body['current_td'] as num?)?.toDouble() ?? metrics.currentTd,
        bestTd: (body['best_td'] as num?)?.toDouble() ?? metrics.bestTd,
        currentScore:
            (body['current_score'] as num?)?.toDouble() ?? metrics.currentScore,
        bestScore:
            (body['best_score'] as num?)?.toDouble() ?? metrics.bestScore,
        iteration: (body['iteration'] as num?)?.toInt() ?? metrics.iteration,
        maxIterations:
            (body['max_iterations'] as num?)?.toInt() ?? metrics.maxIterations,
        logLines: logLines,
        // NEW
        currentAction:
            body['current_action'] as String? ?? metrics.currentAction,
        episodeStep:
            (body['episode_step'] as num?)?.toInt() ?? metrics.episodeStep,
        episodeStepMax:
            (body['episode_step_max'] as num?)?.toInt() ??
            metrics.episodeStepMax,
      );

      if (st == 'complete' || st == 'error' || st == 'stopped') {
        _pollTimer?.cancel();
        status = st == 'complete' ? SolverStatus.complete : SolverStatus.error;

        if (st == 'complete') {
          // Register benchmark result and trigger background fetch.
          // BenchmarkService.setRlResult automatically starts fetching
          // benchmark results regardless of which page the user is on.
          BenchmarkService.instance.setRlResult(
            score: metrics.bestScore,
            nv: metrics.bestNv,
            td: metrics.bestTd,
            jobId: jobId!,
          );

          // Signal that navigation to Solution page is pending.
          // AppShell reads this to decide whether to show a toast (if away)
          // or let SolverScreen start the countdown (if on solver page).
          pendingNavigation = true;
        }
      }

      notifyListeners();
    } catch (_) {
      // Network hiccup — keep polling
    }
  }

  StageStatus _parseStage(String raw) => switch (raw) {
    'done' => StageStatus.done,
    'running' => StageStatus.running,
    'error' => StageStatus.error,
    _ => StageStatus.waiting,
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  /// Called by SolverScreen after it has consumed the pending navigation flag
  /// and begun the countdown toast.
  void clearPendingNavigation() {
    pendingNavigation = false;
    // Don't notify — no UI rebuild needed for this flag reset
  }

  /// Called just before pushing the Solution screen so that SolverScreen can
  /// auto-reset the next time the user returns to it.
  void markSolutionViewed() {
    solutionViewed = true;
    // Don't notify — flag-only update
  }

  /// Called by AppShell when the user taps "View" on the completion toast while
  /// on another page. Re-arms the pending flag so SolverScreen shows the
  /// countdown toast when it mounts.
  void restorePendingNavigation() {
    pendingNavigation = true;
    notifyListeners();
  }

  /// Called by SolutionScreen after successfully fetching solution data.
  /// Caches the raw JSON so navigation back to the page restores the solution.
  void cacheSolution(String jobId, Map<String, dynamic> json) {
    cachedSolutionJobId = jobId;
    cachedSolutionJson = json;
  }

  /// Called by SolverScreen when countdown toast starts.
  /// Locks sidebar Solution nav to prevent manual navigation during countdown.
  void lockSolutionNav() {
    solutionNavLocked = true;
    notifyListeners();
  }

  /// Called by SolverScreen after countdown completes and navigation happens.
  /// Unlocks sidebar Solution nav so user can navigate back if needed.
  void unlockSolutionNav() {
    solutionNavLocked = false;
    // Defer notification until after the current frame to avoid calling
    // setState() during dispose when the widget tree is locked.
    SchedulerBinding.instance.addPostFrameCallback((_) {
      notifyListeners();
    });
  }
}
