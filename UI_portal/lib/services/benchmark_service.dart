import 'dart:async';
import 'dart:convert';

import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/http_utils.dart';
import 'package:RouteIQ_UI/utils/logger.dart';
import 'package:flutter/foundation.dart';

// ─── BENCHMARK SERVICE ────────────────────────────────────────────────────────
//
// Singleton ChangeNotifier that acts as the shared state bridge between
// SolverScreen (writes) and BenchmarkScreen (reads).
//
// Flow:
//   1. User uploads .vrp file in SolverScreen
//      → SolverScreen calls BenchmarkService.instance.setFile(...)
//
//   2. Solve completes (status == "complete") in SolverController
//      → SolverController calls BenchmarkService.instance.setRlResult(...)
//      → BenchmarkService automatically starts fetching benchmark results
//
//   3. User navigates to BenchmarkScreen
//      → BenchmarkScreen reads cached result (no manual fetch needed)
//
// ─────────────────────────────────────────────────────────────────────────────

const _kBaseUrl = ApiConfig.baseUrl;

class BenchmarkService extends ChangeNotifier {
  BenchmarkService._();
  static final instance = BenchmarkService._();

  // ── Uploaded file ─────────────────────────────────────────────────────────
  List<int>? vrpFileBytes;
  String? vrpFileName;

  // ── RL solve result ───────────────────────────────────────────────────────
  // Populated by SolverController when /status returns "complete".
  double? rlScore;
  int? rlNv;
  double? rlTd;
  String? completedJobId;

  // ── Benchmark comparison result ───────────────────────────────────────────
  BenchmarkResult? result;
  bool isRunning = false;
  String? errorMessage;
  int fetchAttempt = 0;

  // ── Setters (all notify listeners) ───────────────────────────────────────

  void setFile(List<int> bytes, String name) {
    vrpFileBytes = bytes;
    vrpFileName = name;
    notifyListeners();
  }

  /// Resets all benchmark results. Called when a new solve starts.
  void resetResults() {
    result = null;
    rlScore = null;
    rlNv = null;
    rlTd = null;
    completedJobId = null;
    errorMessage = null;
    fetchAttempt = 0;
    notifyListeners();
  }

  void setRlResult({
    required double score,
    required int nv,
    required double td,
    required String jobId,
  }) {
    rlScore = score;
    rlNv = nv;
    rlTd = td;
    completedJobId = jobId;
    notifyListeners();

    // Automatically start fetching benchmark results in background
    _fetchBenchmarkResults(jobId);
  }

  void setRunning() {
    isRunning = true;
    errorMessage = null;
    result = null;
    notifyListeners();
  }

  void setBenchmarkResult(BenchmarkResult r) {
    result = r;
    isRunning = false;
    errorMessage = null;
    notifyListeners();
  }

  void setError(String message) {
    isRunning = false;
    errorMessage = message;
    notifyListeners();
  }

  void clearResult() {
    result = null;
    isRunning = false;
    errorMessage = null;
    notifyListeners();
  }

  // ── Background fetch ─────────────────────────────────────────────────────

  /// Fetches benchmark results from the server in background.
  /// Retries on HTTP 425 (still computing) until success or failure.
  Future<void> _fetchBenchmarkResults(String jobId) async {
    isRunning = true;
    errorMessage = null;
    fetchAttempt = 0;
    notifyListeners();

    final url = '$_kBaseUrl/benchmark/$jobId';

    while (true) {
      final res = await safeFetch(url, tag: 'BenchmarkService');
      logDebug(
        'BenchmarkService: fetch attempt ${fetchAttempt + 1} → '
        '${res == null ? 'no response' : 'HTTP ${res.statusCode}'}',
      );

      if (res == null) {
        // Network unreachable or timeout
        errorMessage = 'Could not reach server — check your connection';
        isRunning = false;
        notifyListeners();
        return;
      }

      if (res.statusCode == 200) {
        try {
          final body = jsonDecode(res.body) as Map<String, dynamic>;
          result = BenchmarkResult.fromJson(body);
          isRunning = false;
          errorMessage = null;
          notifyListeners();
          logDebug('BenchmarkService: results loaded successfully');
        } catch (e) {
          logDebug('BenchmarkService: failed to parse response — $e');
          errorMessage = 'Unexpected response format from server';
          isRunning = false;
          notifyListeners();
        }
        return;
      } else if (res.statusCode == 425) {
        // Job still computing — wait and retry
        await Future.delayed(const Duration(seconds: 2));
        fetchAttempt++;
        notifyListeners();
        continue;
      } else {
        logDebug('BenchmarkService: unexpected status ${res.statusCode}');
        errorMessage = 'Failed to load benchmark results (HTTP ${res.statusCode})';
        isRunning = false;
        notifyListeners();
        return;
      }
    }
  }

  /// Manually retry fetching if there was an error.
  void retryFetch() {
    final jobId = completedJobId;
    if (jobId != null) {
      _fetchBenchmarkResults(jobId);
    }
  }

  // ── Convenience getters ───────────────────────────────────────────────────

  bool get hasFile => vrpFileBytes != null && vrpFileName != null;
  bool get hasRlResult => rlScore != null && rlNv != null && rlTd != null;
  bool get hasResult => result != null;
}

// ─── DATA MODELS ─────────────────────────────────────────────────────────────

class SolverComparison {
  final String name;
  final int nv;
  final double td;
  final double score;
  final int? solveTimeSeconds;

  const SolverComparison({
    required this.name,
    required this.nv,
    required this.td,
    required this.score,
    this.solveTimeSeconds,
  });

  factory SolverComparison._fromJson(String name, Map<String, dynamic> j) =>
      SolverComparison(
        name: name,
        nv: (j['nv'] as num).toInt(),
        td: (j['td'] as num).toDouble(),
        score: (j['score'] as num).toDouble(),
        solveTimeSeconds: (j['solve_time_seconds'] as num?)?.toInt(),
      );
}

class BenchmarkResult {
  final String instanceName;
  final List<SolverComparison> comparisons;

  const BenchmarkResult({
    required this.instanceName,
    required this.comparisons,
  });

  /// Parses API response: { instance_name, rl, hgs_default, hgs_large_pop }
  factory BenchmarkResult.fromJson(Map<String, dynamic> j) {
    final comparisons = <SolverComparison>[
      SolverComparison._fromJson('RL', j['rl'] as Map<String, dynamic>),
      SolverComparison._fromJson(
        'HGS Default',
        j['hgs_default'] as Map<String, dynamic>,
      ),
      SolverComparison._fromJson(
        'HGS Large Pop',
        j['hgs_large_pop'] as Map<String, dynamic>,
      ),
    ];
    return BenchmarkResult(
      instanceName: j['instance_name'] as String,
      comparisons: comparisons,
    );
  }

  // ── Convenience getters ─────────────────────────────────────────────────────

  /// RL result (index 0)
  SolverComparison get rl => comparisons[0];

  /// HGS Default result (index 1)
  SolverComparison get hgsDefault => comparisons[1];

  /// HGS Large Pop result (index 2)
  SolverComparison get hgsLargePop => comparisons[2];

  /// % improvement of RL vs HGS Default
  double get pctVsDefault => hgsDefault.score > 0
      ? (hgsDefault.score - rl.score) / hgsDefault.score * 100
      : 0;

  /// % improvement of RL vs HGS Large Pop
  double get pctVsLargePop => hgsLargePop.score > 0
      ? (hgsLargePop.score - rl.score) / hgsLargePop.score * 100
      : 0;

  /// Vehicles saved vs HGS Default
  int get nvSavedVsDefault => hgsDefault.nv - rl.nv;

  /// True if RL beats both baselines
  bool get rlWins => rl.score < hgsDefault.score && rl.score < hgsLargePop.score;

  // Best baseline score (min of non-RL comparisons)
  double get bestBaselineScore =>
      comparisons.skip(1).map((c) => c.score).reduce((a, b) => a < b ? a : b);

  // % improvement of RL vs best baseline
  double get improvementPct =>
      ((bestBaselineScore - rl.score) / bestBaselineScore * 100);

  // Vehicles saved vs best baseline
  int get vehiclesSaved =>
      comparisons.skip(1).map((c) => c.nv).reduce((a, b) => a < b ? a : b) -
      rl.nv;
}
