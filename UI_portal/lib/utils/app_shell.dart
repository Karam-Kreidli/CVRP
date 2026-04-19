import 'package:RouteIQ_UI/components/sidebar.dart';
import 'package:RouteIQ_UI/components/topbar.dart';
import 'package:RouteIQ_UI/screens/benchmark_screen.dart';
import 'package:RouteIQ_UI/screens/dashboard_screen.dart';
import 'package:RouteIQ_UI/screens/pipeline_screen.dart';
import 'package:RouteIQ_UI/screens/placeholder_screen.dart';
import 'package:RouteIQ_UI/screens/solution_screen.dart';
import 'package:RouteIQ_UI/screens/solver_screen.dart';
import 'package:RouteIQ_UI/services/solver_controller.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/http_utils.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'dart:async';
import 'dart:convert';

// ─── API ─────────────────────────────────────────────────────────────────────
const _kBaseUrl = ApiConfig.baseUrl;
// GET /health → { status, device, gpu_name, model_loaded, stage_health, ... }
// ─────────────────────────────────────────────────────────────────────────────

// ─── ROUTE NAMES ─────────────────────────────────────────────────────────────

class RouteName {
  static const dashboard = '/dashboard';
  static const runSolver = '/run';
  static const solution = '/solution';
  static const pipeline = '/pipeline';
  static const benchmark = '/benchmark';

  static const all = <String>[
    dashboard,
    runSolver,
    solution,
    pipeline,
    benchmark,
  ];
}

// ─── HEALTH MODEL ─────────────────────────────────────────────────────────────

enum SystemStatus { loading, ready, error }

class HealthState {
  final SystemStatus status;
  final String device;
  final String gpuName;
  final bool modelLoaded;

  const HealthState({
    this.status = SystemStatus.loading,
    this.device = '—',
    this.gpuName = '—',
    this.modelLoaded = false,
  });

  Color get dotColor => switch (status) {
    SystemStatus.ready => const Color(0xFF00E5A0),
    SystemStatus.loading => const Color(0xFFF0A500),
    SystemStatus.error => const Color(0xFFFF4466),
  };

  String get dotLabel => switch (status) {
    SystemStatus.ready => 'CONNECTED',
    SystemStatus.loading => 'LOADING...',
    SystemStatus.error => 'ERROR',
  };

  bool get isReady => status == SystemStatus.ready;
}

// ─── ROUTE PARSER ─────────────────────────────────────────────────────────────

class AppRouteParser extends RouteInformationParser<String> {
  @override
  Future<String> parseRouteInformation(RouteInformation info) async {
    final path = info.uri.path;
    if (path == '/' || path.isEmpty) return RouteName.dashboard;
    if (RouteName.all.contains(path)) return path;
    return RouteName.dashboard;
  }

  @override
  RouteInformation? restoreRouteInformation(String config) =>
      RouteInformation(uri: Uri.parse(config));
}

// ─── ROUTER DELEGATE ─────────────────────────────────────────────────────────

class AppRouterDelegate extends RouterDelegate<String> with ChangeNotifier {
  String _currentRoute = RouteName.dashboard;
  int _generation = 0;
  String? _pendingJobId;
  String? _lastCompletedJobId;

  String get currentRoute => _currentRoute;
  int get generation => _generation;
  String? get pendingJobId => _pendingJobId;
  String? get lastCompletedJobId => _lastCompletedJobId;

  @override
  String get currentConfiguration => _currentRoute;

  void push(String route) {
    _currentRoute = route;
    _pendingJobId = null;
    _generation++;
    notifyListeners();
  }

  void pushSolution({required String jobId}) {
    _currentRoute = RouteName.solution;
    _pendingJobId = jobId;
    _lastCompletedJobId = jobId;
    _generation++;
    notifyListeners();
  }

  void pushBenchmark() {
    _currentRoute = RouteName.benchmark;
    _pendingJobId = null;
    _generation++;
    notifyListeners();
  }

  @override
  Future<bool> popRoute() async => false;

  @override
  Future<void> setNewRoutePath(String config) async {
    if (_currentRoute != config) {
      _currentRoute = config;
      _pendingJobId = null;
      notifyListeners();
    }
  }

  @override
  Widget build(BuildContext context) => Navigator(
    onPopPage: (route, result) => route.didPop(result),
    pages: [
      MaterialPage(
        key: const ValueKey('shell'),
        child: _AppShell(delegate: this),
      ),
    ],
  );
}

// ─── GLOBAL SINGLETONS ───────────────────────────────────────────────────────

final shellRouter = AppRouterDelegate();
final shellRouteParser = AppRouteParser();

class ShellNav {
  static void push(String route) => shellRouter.push(route);
  static void pushSolution({required String jobId}) =>
      shellRouter.pushSolution(jobId: jobId);
  static void pushBenchmark() => shellRouter.pushBenchmark();
}

// ─── APP SHELL ────────────────────────────────────────────────────────────────
//
// Responsibilities:
//  1. Health polling (GET /health every 15s) — single source of truth
//  2. Passes HealthState to topbar, sidebar, dashboard
//  3. Listens to SolverController — shows a top overlay toast when the solver
//     completes while the user is NOT on the Solver Console page.
//     The toast matches the old solver screen's countdown toast design but
//     has a "View" button instead of a timer. Pressing "View":
//       a. dismisses the toast
//       b. calls ctrl.restorePendingNavigation() so SolverScreen shows the
//          countdown toast when it mounts
//       c. navigates to the Solver Console page

class _AppShell extends StatefulWidget {
  final AppRouterDelegate delegate;
  const _AppShell({required this.delegate});

  @override
  State<_AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<_AppShell> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  HealthState _health = const HealthState(status: SystemStatus.loading);
  Timer? _healthTimer;

  // ── Completion overlay (top toast shown when solve finishes away from solver)
  OverlayEntry? _completionOverlay;

  AppRouterDelegate get _delegate => widget.delegate;
  String get _active => _delegate.currentRoute;

  // Stored so the overlay callback can call Overlay.of() without needing build ctx
  BuildContext? _shellContext;

  @override
  void initState() {
    super.initState();
    _delegate.addListener(_onRouteChanged);
    SolverController.instance.addListener(_onSolverChanged);
    _fetchHealth();
    _healthTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _fetchHealth(),
    );
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _delegate.removeListener(_onRouteChanged);
    SolverController.instance.removeListener(_onSolverChanged);
    _dismissCompletionOverlay();
    super.dispose();
  }

  void _onRouteChanged() {
    if (!mounted) return;

    // If user navigates to solver page while completion overlay is showing,
    // dismiss it and restore pending navigation so the countdown toast appears
    // in SolverScreen (same behavior as when user was on solver page from start).
    if (_active == RouteName.runSolver && _completionOverlay != null) {
      _dismissCompletionOverlay();
      SolverController.instance.restorePendingNavigation();
    }

    setState(() {});
  }

  // ── Solver completion handler ─────────────────────────────────────────────
  //
  // Fires every time SolverController.notifyListeners() is called.
  // We only act when:
  //   • pendingNavigation was just set (solve just completed)
  //   • user is NOT currently on the Solver Console page
  //
  // If the user IS on the Solver Console page, SolverScreen handles this
  // itself via its own listener and shows the countdown toast there.

  void _onSolverChanged() {
    if (!mounted) return;
    final ctrl = SolverController.instance;

    if (ctrl.pendingNavigation && _active != RouteName.runSolver) {
      // User is on Dashboard / Solution / Pipeline / Benchmark while solve
      // finished. Show a "View" toast at the top of the screen.
      // Mark handled immediately so we don't re-show on the next notify.
      ctrl.clearPendingNavigation();
      _showCompletionOverlay();
    }

    // Always rebuild so sidebar reflects solutionNavLocked state
    setState(() {});
  }

  // ── Completion toast — top-of-screen overlay matching old solver screen ────

  void _showCompletionOverlay() {
    _dismissCompletionOverlay();
    final ctx = _shellContext;
    if (ctx == null) return;

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
                            text: '  —  ready to view results',
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

                  // "View" button — navigates to Solver Console + re-arms countdown
                  MouseRegion(
                    cursor: SystemMouseCursors.click,
                    child: GestureDetector(
                      onTap: () {
                        _dismissCompletionOverlay();
                        // Re-arm so SolverScreen shows the countdown toast on mount
                        SolverController.instance.restorePendingNavigation();
                        _delegate.push(RouteName.runSolver);
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 14,
                          vertical: 7,
                        ),
                        decoration: BoxDecoration(
                          color: AppColors.green.withOpacity(0.12),
                          border: Border.all(
                            color: AppColors.green.withOpacity(0.5),
                            width: 1.5,
                          ),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          'View',
                          style: GoogleFonts.syne(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: AppColors.green,
                          ),
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

    _completionOverlay = entry;
    Overlay.of(ctx).insert(entry);
  }

  void _dismissCompletionOverlay() {
    _completionOverlay?.remove();
    _completionOverlay = null;
  }

  // ── Health ────────────────────────────────────────────────────────────────

  Future<void> _fetchHealth() async {
    final res = await safeGet(
      '$_kBaseUrl/health',
      tag: 'AppShell',
      timeout: const Duration(seconds: 5),
    );

    if (!mounted) return;

    if (res != null && res.statusCode == 200) {
      try {
        final body = jsonDecode(res.body) as Map<String, dynamic>;
        final raw = body['status'] as String? ?? 'error';
        setState(() {
          _health = HealthState(
            status: switch (raw) {
              'ready' => SystemStatus.ready,
              'loading' => SystemStatus.loading,
              _ => SystemStatus.error,
            },
            device: body['device'] as String? ?? '—',
            gpuName: body['gpu_name'] as String? ?? '—',
            modelLoaded: body['model_loaded'] as bool? ?? false,
          );
        });
      } catch (_) {
        setState(() => _health = const HealthState(status: SystemStatus.error));
      }
    } else {
      setState(() => _health = const HealthState(status: SystemStatus.error));
    }
  }

  void _navigate(String route) {
    _delegate.push(route);
    final s = _scaffoldKey.currentState;
    if (s != null && s.isDrawerOpen) s.closeDrawer();
  }

  Widget _buildPage() {
    switch (_active) {
      case RouteName.runSolver:
        return const SolverScreen();

      case RouteName.solution:
        return SolutionScreen(jobId: _delegate.pendingJobId);

      case RouteName.pipeline:
        return PipelineScreen(health: _health);

      case RouteName.benchmark:
        return BenchmarkScreen();

      case RouteName.dashboard:
      default:
        return DashboardScreen(health: _health);
    }
  }

  bool _isSmall(BuildContext ctx) => MediaQuery.of(ctx).size.width < 900;

  @override
  Widget build(BuildContext context) {
    // Store context so the overlay callback can call Overlay.of() without
    // needing to pass context through the listener chain.
    _shellContext = context;

    final small = _isSmall(context);
    final sidebar = AppSidebar(
      activeRoute: _active,
      onNavigate: _navigate,
      health: _health,
      solutionNavLocked: SolverController.instance.solutionNavLocked,
    );
    final topbar = AppTopBar(activeRoute: _active, health: _health);

    final content = AnimatedSwitcher(
      duration: const Duration(milliseconds: 180),
      switchInCurve: Curves.easeOut,
      transitionBuilder: (child, anim) => FadeTransition(
        opacity: anim,
        child: SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0.015, 0),
            end: Offset.zero,
          ).animate(anim),
          child: child,
        ),
      ),
      child: KeyedSubtree(
        key: ValueKey('${_active}_${_delegate.generation}'),
        child: _buildPage(),
      ),
    );

    return small
        ? Scaffold(
            key: _scaffoldKey,
            backgroundColor: const Color(0xFF050810),
            drawer: Drawer(
              width: 220,
              backgroundColor: Colors.transparent,
              child: sidebar,
            ),
            body: Column(
              children: [
                Stack(
                  children: [
                    topbar,
                    Positioned(
                      left: 12,
                      top: 0,
                      bottom: 0,
                      child: Center(
                        child: IconButton(
                          icon: const Icon(
                            Icons.menu_rounded,
                            color: Color(0xFF6B82A8),
                            size: 20,
                          ),
                          onPressed: () =>
                              _scaffoldKey.currentState?.openDrawer(),
                        ),
                      ),
                    ),
                  ],
                ),
                Expanded(child: content),
              ],
            ),
          )
        : Scaffold(
            backgroundColor: const Color(0xFF050810),
            body: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                sidebar,
                Expanded(
                  child: Column(
                    children: [
                      topbar,
                      Expanded(child: content),
                    ],
                  ),
                ),
              ],
            ),
          );
  }
}
