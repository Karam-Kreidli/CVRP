import 'package:RouteIQ_UI/screens/dashboard_screen.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/widgets/Tag.dart';
import 'package:RouteIQ_UI/widgets/animatedDot.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';



// ─── SIDEBAR ──────────────────────────────────────────────────────────────────
//
// Health state is passed in from _AppShell (single source of truth).
// AppShell owns the polling and passes the HealthState down here. This guarantees the
// sidebar dot and topbar dot are always identical.
//
// ─────────────────────────────────────────────────────────────────────────────

// ─── NAV ITEM MODEL ───────────────────────────────────────────────────────────

class _NavItem {
  final String label;
  final IconData icon;
  final String route;
  const _NavItem(this.icon, this.label, this.route);
}

const _navItems = [
  _NavItem(Icons.dashboard, 'Dashboard', RouteName.dashboard),
  _NavItem(Icons.play_arrow, 'Run Solver', RouteName.runSolver),
  _NavItem(Icons.lightbulb, 'Solution', RouteName.solution),
  _NavItem(Icons.bar_chart, 'Benchmark', RouteName.benchmark),
  _NavItem(Icons.account_tree, 'Pipeline', RouteName.pipeline),
  
];

const _teamMembers = [
  'AHMED RAHIL',
  'KARAM BERHAN',
  'MOHAMMED ABDUL HARIS',
  'MOHAMMED BIN ALI MAQQAVI',
  'MUHAMMED NIHAL',
];

// ─── SIDEBAR ──────────────────────────────────────────────────────────────────

class AppSidebar extends StatelessWidget {
  final String activeRoute;
  final ValueChanged<String> onNavigate;
  final HealthState health;
  final bool solutionNavLocked;

  const AppSidebar({
    super.key,
    required this.activeRoute,
    required this.onNavigate,
    required this.health,
    this.solutionNavLocked = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 220,
      color: AppColors.bg1,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(height: 1, color: AppColors.border),
          _buildLogo(),
          _buildDivider(),
          _buildStatusPill(context),
          _buildDivider(),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 6),
            child: Text('NAVIGATE', style: AppTextStyles.navSectionLabel),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: _navItems
                    .map(
                      (item) => _SidebarNavItem(
                        item: item,
                        isActive: activeRoute == item.route,
                        onTap: () => onNavigate(item.route),
                        isDisabled: item.route != RouteName.runSolver && solutionNavLocked,
                      ),
                    )
                    .toList(),
              ),
            ),
          ),
          // _buildDivider(),
          // _buildBottomInfo(),
          _buildDivider(),
          GestureDetector(
            onTap: () => _showTeamDialog(context),
            child: MouseRegion(
              cursor: SystemMouseCursors.click,
              child: _buildUserTile(),
            ),
          ),
        ],
      ),
    );
  }

  // ── Logo ──────────────────────────────────────────────────────────────────

  Widget _buildLogo() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 22, 20, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // ── Swap for your logo asset when available:
          Container(
            width: 200, height: 50,
            decoration: const BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/images/routeIQ_logo_white.png'),
              ),
            ),
          ),
          // RichText(
          //   text: TextSpan(
          //     children: [
          //       TextSpan(
          //         text: 'ML',
          //         style: GoogleFonts.syne(
          //           fontSize: 18,
          //           fontWeight: FontWeight.w800,
          //           color: AppColors.textPrimary,
          //           letterSpacing: 0.02,
          //         ),
          //       ),
          //       TextSpan(
          //         text: '4',
          //         style: GoogleFonts.syne(
          //           fontSize: 18,
          //           fontWeight: FontWeight.w800,
          //           color: AppColors.cyan,
          //         ),
          //       ),
          //       TextSpan(
          //         text: 'VRP',
          //         style: GoogleFonts.syne(
          //           fontSize: 18,
          //           fontWeight: FontWeight.w800,
          //           color: AppColors.textPrimary,
          //         ),
          //       ),
          //     ],
          //   ),
          // ),
          const SizedBox(height: 4),
          Text(
            'GECCO 2026 · CVRP TRACK',
            style: AppTextStyles.monoLabel.copyWith(fontSize: 9),
          ),
        ],
      ),
    );
  }

  // ── Status Pill ───────────────────────────────────────────────────────────

  Widget _buildStatusPill(BuildContext context) {
    final color = health.dotColor;
    final label = health.dotLabel;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedStatusDot(color: color),
          const SizedBox(width: 8),
          Text(label, style: AppTextStyles.monoLabel.copyWith(color: color)),
          // Device badge — only shown once backend is connected
          if (health.status == SystemStatus.ready && health.device != '—') ...[
            const SizedBox(width: 6),
            // Container(
            //   padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
            //   decoration: BoxDecoration(
            //     color: color.withOpacity(0.12),
            //     border: Border.all(color: color.withOpacity(0.3)),
            //     borderRadius: BorderRadius.circular(3),
            //   ),
            //   child: Text(
            //     health.device.toUpperCase(),
            //     style: AppTextStyles.monoLabel.copyWith(
            //       fontSize: 8,
            //       color: color,
            //     ),
            //   ),
            // ),
            // Tag(
            //   label: health.device.toUpperCase(),
            //   color: AppColors.green,
            //   fixWidth: false,)
          ],
        ],
      ),
    );
  }

  // ── Bottom Info ───────────────────────────────────────────────────────────

  Widget _buildBottomInfo() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'X-Dataset · 100–400 nodes',
            style: AppTextStyles.monoLabel.copyWith(fontSize: 9),
          ),
          const SizedBox(height: 3),
          Text(
            'Deadline: 13 Jun 2026',
            style: AppTextStyles.monoLabel.copyWith(
              color: AppColors.amberDim,
              fontSize: 9,
            ),
          ),
        ],
      ),
    );
  }

  // ── User Tile ─────────────────────────────────────────────────────────────

  Widget _buildUserTile() {
    return Container(
      margin: const EdgeInsets.all(10),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.bg3,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 15,
            backgroundColor: AppColors.cyan.withOpacity(0.2),
            child: const Icon(Icons.groups, size: 16, color: AppColors.cyan),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Team RouteIQ',
                  style: AppTextStyles.navLabelActive.copyWith(fontSize: 11),
                ),
                Text(
                  '5 members',
                  style: AppTextStyles.monoLabel.copyWith(
                    fontSize: 9,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
          // const Text(
          //   '⋮',
          //   style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
          // ),
        ],
      ),
    );
  }

  // ── Team Dialog ───────────────────────────────────────────────────────────

  void _showTeamDialog(BuildContext context) {
    final color = health.dotColor;
    final label = health.dotLabel;

    showDialog(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: AppColors.bg3,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
          side: const BorderSide(color: AppColors.border),
        ),
        child: SizedBox(
          width: 400,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: AppColors.cyan,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.groups,
                        color: Colors.black,
                        size: 18,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'RouteIQ v1.0.0',
                          style: GoogleFonts.syne(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          'Meet the team',
                          style: AppTextStyles.monoSmall.copyWith(
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ],
                    ),
                    const Spacer(),
                    
                  ],
                ),

                const SizedBox(height: 14),
                Divider(color: Colors.white.withOpacity(0.08), height: 1),
                const SizedBox(height: 14),

                ..._teamMembers.map(
                  (member) => Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      children: [
                        CircleAvatar(
                          radius: 16,
                          backgroundColor: AppColors.cyan,
                          child: Text(
                            member.split(' ').map((w) => w[0]).take(2).join(),
                            style: const TextStyle(
                              color: Colors.black,
                              fontSize: 10,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            member,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                // GPU / device row — only visible once backend is connected
                
                  Divider(color: Colors.white.withOpacity(0.08), height: 1),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Text('Backend · ', style: AppTextStyles.monoSmall),
                      if(health.device != '—')...[
                        Text(
                        health.gpuName,
                        style: AppTextStyles.monoSmall.copyWith(
                          color: AppColors.cyan,
                        ),
                      ),
                      const SizedBox(width: 8),
                      // Container(
                      //   padding: const EdgeInsets.symmetric(
                      //     horizontal: 6,
                      //     vertical: 2,
                      //   ),
                      //   decoration: BoxDecoration(
                      //     color: AppColors.green.withOpacity(0.12),
                      //     border: Border.all(
                      //       color: AppColors.green.withOpacity(0.3),
                      //     ),
                      //     borderRadius: BorderRadius.circular(3),
                      //   ),
                      //   child: Text(
                      //     health.device.toUpperCase(),
                      //     style: AppTextStyles.monoLabel.copyWith(
                      //       color: AppColors.green,
                      //       fontSize: 8,
                      //     ),
                      //   ),
                      // ),
                      Tag(label: health.device.toUpperCase(), color: AppColors.green, fixWidth: false)
                      ],
                      const SizedBox(width: 5),
                      // Live status chip — mirrors the sidebar pill
                      Container(
                        // height: 40,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: color.withOpacity(0.12),
                          border: Border.all(color: color.withOpacity(0.4)),
                          borderRadius: BorderRadius.circular(3),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            AnimatedStatusDot(color: color),
                            const SizedBox(width: 5),
                            Text(
                              label,
                              style: AppTextStyles.monoLabel.copyWith(
                                color: color,
                                fontSize: 9,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
              

                const SizedBox(height: 4),
                SizedBox(
                  width: double.infinity,
                  child: TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    style: TextButton.styleFrom(
                      backgroundColor: AppColors.cyan.withOpacity(0.08),
                      padding: const EdgeInsets.symmetric(vertical: 13),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                        side: BorderSide(
                          color: AppColors.cyan.withOpacity(0.5),
                        ),
                      ),
                    ),
                    child: Text(
                      'Close',
                      style: GoogleFonts.syne(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: AppColors.cyan,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDivider() => Container(height: 1, color: AppColors.border);
}

// ─── SIDEBAR NAV ITEM ─────────────────────────────────────────────────────────

class _SidebarNavItem extends StatefulWidget {
  final _NavItem item;
  final bool isActive;
  final VoidCallback onTap;
  final bool isDisabled;

  const _SidebarNavItem({
    required this.item,
    required this.isActive,
    required this.onTap,
    this.isDisabled = false,
  });

  @override
  State<_SidebarNavItem> createState() => _SidebarNavItemState();
}

class _SidebarNavItemState extends State<_SidebarNavItem> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final active = widget.isActive;
    final disabled = widget.isDisabled;
    return MouseRegion(
      cursor: disabled ? SystemMouseCursors.forbidden : SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: disabled ? null : widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 1),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
          decoration: BoxDecoration(
            color: disabled
                ? Colors.transparent
                : active
                    ? AppColors.cyanFaint
                    : _hovered
                        ? AppColors.cyanFaint.withOpacity(0.5)
                        : Colors.transparent,
            borderRadius: BorderRadius.circular(6),
            border: Border(
              left: BorderSide(
                color: disabled
                    ? Colors.transparent
                    : active
                        ? AppColors.cyan
                        : _hovered
                            ? AppColors.cyanDim
                            : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Row(
            children: [
              Icon(
                widget.item.icon,
                size: 18,
                color: disabled
                    ? AppColors.textMuted
                    : active || _hovered
                        ? AppColors.cyan
                        : AppColors.textSecondary,
              ),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  widget.item.label,
                  style: disabled
                      ? AppTextStyles.navLabel.copyWith(color: AppColors.textMuted)
                      : active || _hovered
                          ? AppTextStyles.navLabelActive
                          : AppTextStyles.navLabel,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── PULSING DOT ─────────────────────────────────────────────────────────────

// class _PulsingDot extends StatefulWidget {
//   final Color color;
//   const _PulsingDot({required this.color});

//   @override
//   State<_PulsingDot> createState() => _PulsingDotState();
// }

// class _PulsingDotState extends State<_PulsingDot>
//     with SingleTickerProviderStateMixin {
//   late final AnimationController _ctrl;
//   late final Animation<double> _anim;

//   @override
//   void initState() {
//     super.initState();
//     _ctrl = AnimationController(
//       vsync: this,
//       duration: const Duration(seconds: 2),
//     )..repeat(reverse: true);
//     _anim = Tween<double>(begin: 1.0, end: 0.3).animate(_ctrl);
//   }

//   @override
//   void dispose() {
//     _ctrl.dispose();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) => FadeTransition(
//     opacity: _anim,
//     child: Container(
//       width: 6,
//       height: 6,
//       decoration: BoxDecoration(color: widget.color, shape: BoxShape.circle),
//     ),
//   );
// }

// ─── GOOGLE FONTS HELPER ─────────────────────────────────────────────────────

class GoogleFontsHelper {
  static TextStyle syneBold({
    required double fontSize,
    required Color color,
    double? letterSpacing,
  }) => GoogleFonts.syne(
    fontSize: fontSize,
    fontWeight: FontWeight.w800,
    color: color,
    letterSpacing: letterSpacing,
  );
}
