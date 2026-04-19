import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/widgets/animatedDot.dart';
import 'package:flutter/material.dart';

// ─── TOP BAR ─────────────────────────────────────────────────────────────────
//
// Receives HealthState from _AppShell (the single source of truth).
// The pulsing dot colour reflects the live backend status:
//   Green  → backend ready   ("SYSTEM READY")
//   Amber  → backend loading ("CONNECTING...")
//   Red    → backend error   ("ERROR")
//
// The topbar dot and the sidebar dot are always in sync because they both
// receive the same HealthState object from _AppShell — no separate HTTP call.
//
// ─────────────────────────────────────────────────────────────────────────────

String _maskUrl(String url, {int headLength = 15, int tailLength = 12}) {
  if (url.length <= headLength + tailLength + 6) return url;
  final head = url.substring(0, headLength);
  final tail = url.substring(url.length - tailLength);
  return '$head****$tail';
}

class AppTopBar extends StatelessWidget implements PreferredSizeWidget {
  final String activeRoute;
  final HealthState health;

  const AppTopBar({super.key, required this.activeRoute, required this.health});

  @override
  Size get preferredSize => const Size.fromHeight(40);

  String get _pageLabel => switch (activeRoute) {
    RouteName.dashboard => 'Dashboard',
    RouteName.runSolver => 'Solver Console',
    RouteName.solution => 'Solution Viewer',
    RouteName.pipeline => 'Pipeline Overview',
    _ => 'Dashboard',
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 40,
      color: AppColors.bg1,
      child: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final isCompact = constraints.maxWidth < 500;
                  return Row(
                    children: [
                      // Left: current page label
                      // _PulsingDot(color: AppColors.textMuted),
                      SizedBox(width: isCompact ? 35 : 8),
                      // Text(
                      //   _pageLabel,
                      //   style: AppTextStyles.monoSmall.copyWith(
                      //     color: AppColors.textSecondary,
                      //   ),
                      // ),

                      const Spacer(),

                      // // Right: meta tags
                      // _MetaTag(
                      //   label: 'Track',
                      //   value: 'CVRP',
                      //   valueColor: AppColors.cyan,
                      //   compact: isCompact,
                      // ),
                      // SizedBox(width: isCompact ? 10 : 16),
                      // _MetaTag(
                      //   label: 'Dataset',
                      //   value: 'X-instances',
                      //   valueColor: AppColors.textSecondary,
                      //   compact: isCompact,
                      // ),
                      // SizedBox(width: isCompact ? 10 : 16),
                      _MetaTag(
                        label: 'API',
                        value: _maskUrl(ApiConfig.baseUrl),
                        // value: "http://localhost:8080",
                        // API value colour reflects health status:
                        //   green = backend responding
                        //   amber = backend loading
                        //   red   = backend unreachable
                        valueColor: health.dotColor,
                        compact: isCompact,
                        maxWidth: isCompact ? 100 : 1800,
                      ),
                      const SizedBox(width: 10),

                      // The status dot — driven by HealthState from AppShell
                    ],
                  );
                },
              ),
            ),
          ),
          Container(height: 1, color: AppColors.border),
        ],
      ),
    );
  }
}

// ─── META TAG ─────────────────────────────────────────────────────────────────

class _MetaTag extends StatelessWidget {
  final String label;
  final String value;
  final Color valueColor;
  final bool compact;
  final double? maxWidth;

  const _MetaTag({
    required this.label,
    required this.value,
    required this.valueColor,
    this.compact = false,
    this.maxWidth,
  });

  @override
  Widget build(BuildContext context) {
    if (compact) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label:', style: AppTextStyles.monoSmall.copyWith(fontSize: 9)),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              ConstrainedBox(
                constraints: BoxConstraints(maxWidth: maxWidth ?? double.infinity),
                child: Text(
                  value,
                  style: AppTextStyles.monoSmall.copyWith(
                    color: valueColor,
                    fontWeight: FontWeight.w500,
                    fontSize: 10,
                  ),
                  overflow: TextOverflow.ellipsis,
                  maxLines: 1,
                ),
              ),
              if (label == 'API') ...[
                const SizedBox(width: 4),
                AnimatedStatusDot(color: valueColor),
              ],
            ],
          ),
        ],
      );
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('$label: ', style: AppTextStyles.monoSmall),
        ConstrainedBox(
          constraints: BoxConstraints(maxWidth: maxWidth ?? double.infinity),
          child: Text(
            value,
            style: AppTextStyles.monoSmall.copyWith(
              color: valueColor,
              fontWeight: FontWeight.w500,
            ),
            overflow: TextOverflow.ellipsis,
            maxLines: 1,
          ),
        ),
        if (label == 'API') ...[
          const SizedBox(width: 6),
          AnimatedStatusDot(color: valueColor),
        ],
      ],
    );
  }
}


