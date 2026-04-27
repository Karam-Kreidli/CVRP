// ignore_for_file: unused_import
import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:RouteIQ_UI/models/data.dart';
import 'package:RouteIQ_UI/screens/dashboard_screen.dart';
import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:RouteIQ_UI/utils/api_config.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:RouteIQ_UI/utils/http_utils.dart';
import 'package:RouteIQ_UI/utils/logger.dart';
import 'package:RouteIQ_UI/widgets/Tag.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ─── API INTEGRATION ─────────────────────────────────────────────────────────
//
// Endpoint: GET /health  → { status, device, gpu_name, model_loaded }
// Endpoint: GET /runs?limit=1000 → { total, runs: [...] }
//
// ─────────────────────────────────────────────────────────────────────────────

// ─── CHANGE: Updated stage data for 4-stage architecture ─────────────────────
//
// Old 5 stages:  GNN Observer → Fleet Manager → HGS Engine → Route Driver → MACA Trainer
// New 4 stages:  Feature Extractor → Fleet Manager → HGS Engine → PPO Trainer
//
// Rationale (from src/solver_engine.py + src/agent_manager.py):
//   Stage 1 — GNN Observer replaced by hand-crafted feature extraction.
//              7 deterministic features computed once per episode; no network.
//   Stage 2 — Fleet Manager unchanged: Actor-Critic PPO, 14-dim obs, 7 actions.
//   Stage 3 — HGS Engine unchanged: hygese C++ solver, 500–1500 iter/step.
//   Stage 4 — Route Driver and MACA Trainer replaced by a single PPO Trainer
//              that updates the Fleet Manager policy via GAE-λ advantages.
//
// NOTE: This list is defined locally to override whatever is in data.dart.
//       If data.dart also exports a `stages` variable, rename this one or
//       remove the export from data.dart to avoid shadowing.


// ─── PIPELINE SCREEN ─────────────────────────────────────────────────────────

class PipelineScreen extends StatefulWidget {
  const PipelineScreen({super.key, required this.health});

  final HealthState health;

  @override
  State<PipelineScreen> createState() => _PipelineScreenState();
}

class _PipelineScreenState extends State<PipelineScreen> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg0,
      child: Stack(
        children: [
          const _GridBackground(),
          SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(32, 28, 32, 60),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),
                const SizedBox(height: 24),
                _buildFlowDiagram(),
                const SizedBox(height: 24),
                ...stages.map(
                  (s) => Padding(
                    padding: const EdgeInsets.only(bottom: 14),
                    child: _StageCard(stage: s),
                  ),
                ),
                const SizedBox(height: 8),
                _buildBottomBar(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── CHANGE: updated from "5-Stage Pipeline" to "4-Stage Pipeline" ────
        Text('4-Stage Pipeline', style: AppTextStyles.displayMedium),
        const SizedBox(height: 6),
      ],
    );
  }

  // ── Flow Diagram ──────────────────────────────────────────────────────────

  Widget _buildFlowDiagram() {
    // ── CHANGE: 4 nodes replacing the old 5 ──────────────────────────────────
    //
    // Removed: GNN (cyan), Route Driver (green), MACA Trainer (orange)
    // Added:   Feature Extractor (cyan), PPO Trainer (green)
    // Kept:    Fleet Manager (amber), HGS Engine (purple)

    final nodes = [
      ('Feature\nExtractor', AppColors.cyan),
      ('Fleet\nManager', AppColors.amber),
      ('HGS\nEngine', AppColors.purple),
      ('PPO\nTrainer', AppColors.green),
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final isSmall = constraints.maxWidth < 700;

        final objectiveWidget = Column(
          crossAxisAlignment: isSmall
              ? CrossAxisAlignment.center
              : CrossAxisAlignment.start,
          children: [
            Text(
              'Objective:',
              style: AppTextStyles.monoLabel.copyWith(
                color: AppColors.textMuted,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'min(1000×NV + TD)',
              style: AppTextStyles.mono.copyWith(
                color: AppColors.amber,
                fontSize: 12,
              ),
            ),
          ],
        );

        if (isSmall) {
          return SizedBox(
            width: double.infinity,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
              decoration: BoxDecoration(
                color: AppColors.bg2,
                border: Border.all(color: AppColors.border),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                children: [
                  ...List.generate(nodes.length, (i) {
                    final (label, color) = nodes[i];
                    return Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _FlowNode(label: label, color: color),
                        if (i < nodes.length - 1)
                          _FlowArrowVertical(
                            fromColor: color,
                            toColor: nodes[i + 1].$2,
                          ),
                      ],
                    );
                  }),
                  const SizedBox(height: 16),
                  Container(height: 1, width: 80, color: AppColors.border),
                  const SizedBox(height: 16),
                  objectiveWidget,
                ],
              ),
            ),
          );
        }

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
          decoration: BoxDecoration(
            color: AppColors.bg2,
            border: Border.all(color: AppColors.border),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Row(
            children: [
              Expanded(
                child: Center(
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        ...List.generate(nodes.length, (i) {
                          final (label, color) = nodes[i];
                          return Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _FlowNode(label: label, color: color),
                              if (i < nodes.length - 1)
                                _FlowArrow(
                                  fromColor: color,
                                  toColor: nodes[i + 1].$2,
                                ),
                            ],
                          );
                        }),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Container(width: 1, height: 48, color: AppColors.border),
              const SizedBox(width: 16),
              objectiveWidget,
            ],
          ),
        );
      },
    );
  }

  // ── Bottom Bar ────────────────────────────────────────────────────────────

  Widget _buildBottomBar() {
    // ── CHANGE: updated tech tags ─────────────────────────────────────────────
    //
    // Removed: PyG (no GNN — graph neural network is gone)
    // Added:   hygese (the HGS-CVRP Python binding used in solver_engine.py)
    // Kept:    PyTorch, FastAPI, Flutter

    final techTags = Wrap(
      spacing: 6,
      runSpacing: 6,
      children: [
        Tag(label: 'hygese', color: AppColors.cyan, fixWidth: false),
        Tag(label: 'PyTorch', color: AppColors.amber, fixWidth: false),
        Tag(label: 'FastAPI', color: AppColors.green, fixWidth: false),
        Tag(label: 'Flutter', color: AppColors.orange, fixWidth: false),
      ],
    );

    final labelContent = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'GECCO 2026 · ML4VRP Competition',
          style: AppTextStyles.heading.copyWith(fontSize: 13),
        ),
        const SizedBox(height: 3),
      ],
    );

    return LayoutBuilder(
      builder: (context, constraints) {
        final isSmall = constraints.maxWidth < 500;

        return Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
          decoration: BoxDecoration(
            color: AppColors.bg2,
            border: Border.all(color: AppColors.border),
            borderRadius: BorderRadius.circular(10),
          ),
          child: isSmall
              ? Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    labelContent,
                    const SizedBox(height: 12),
                    techTags,
                  ],
                )
              : Row(
                  children: [
                    Expanded(child: labelContent),
                    const SizedBox(width: 20),
                    techTags,
                  ],
                ),
        );
      },
    );
  }
}

// ─── FLOW NODE ────────────────────────────────────────────────────────────────

class _FlowNode extends StatefulWidget {
  final String label;
  final Color color;
  const _FlowNode({required this.label, required this.color});

  @override
  State<_FlowNode> createState() => _FlowNodeState();
}

class _FlowNodeState extends State<_FlowNode> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        constraints: const BoxConstraints(minWidth: 72),
        decoration: BoxDecoration(
          color: _hovered
              ? widget.color.withOpacity(0.2)
              : widget.color.withOpacity(0.1),
          border: Border.all(
            color: _hovered
                ? widget.color.withOpacity(0.8)
                : widget.color.withOpacity(0.45),
          ),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          widget.label,
          textAlign: TextAlign.center,
          style: GoogleFonts.blinker(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: widget.color,
            height: 1.3,
          ),
        ),
      ),
    );
  }
}

// ─── FLOW ARROW ───────────────────────────────────────────────────────────────

class _FlowArrow extends StatelessWidget {
  final Color fromColor;
  final Color toColor;
  const _FlowArrow({required this.fromColor, required this.toColor});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 28,
      child: Row(
        children: [const SizedBox(width: 2), const Icon(Icons.arrow_right)],
      ),
    );
  }
}

// ─── FLOW ARROW VERTICAL ──────────────────────────────────────────────────────

class _FlowArrowVertical extends StatelessWidget {
  final Color fromColor;
  final Color toColor;
  const _FlowArrowVertical({required this.fromColor, required this.toColor});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 28,
      child: Column(
        children: [const SizedBox(width: 2), const Icon(Icons.arrow_drop_down)],
      ),
    );
  }
}

// ─── STAGE CARD ───────────────────────────────────────────────────────────────

class _StageCard extends StatefulWidget {
  final Stage stage;
  const _StageCard({required this.stage});

  @override
  State<_StageCard> createState() => _StageCardState();
}

class _StageCardState extends State<_StageCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final s = widget.stage;

    return Stack(
      children: [
        AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            color: AppColors.bg2,
            border: Border.all(color: AppColors.border),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Card header ───────────────────────────────────────────────
              LayoutBuilder(
                builder: (context, constraints) {
                  final isSmallCard = constraints.maxWidth < 500;

                  return Padding(
                    padding: const EdgeInsets.fromLTRB(20, 18, 20, 0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Stage number badge
                        Container(
                          width: 38,
                          height: 38,
                          decoration: BoxDecoration(
                            color: s.accentColor.withOpacity(0.15),
                            border: Border.all(
                              color: s.accentColor.withOpacity(0.5),
                            ),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Center(
                            child: Text(
                              '${s.num}',
                              style: GoogleFonts.blinker(
                                fontSize: 20,
                                fontWeight: FontWeight.w800,
                                color: s.accentColor,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 14),

                        // Name + model tag
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              if (isSmallCard) ...[
                                Text(s.name, style: AppTextStyles.heading),
                                const SizedBox(height: 6),
                                Tag(
                                  label: s.modelTag,
                                  color: s.accentColor,
                                  fixWidth: false,
                                ),
                              ] else ...[
                                Row(
                                  children: [
                                    Flexible(
                                      child: Text(
                                        s.name,
                                        style: AppTextStyles.heading,
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Tag(
                                      label: s.modelTag,
                                      color: s.accentColor,
                                      fixWidth: false,
                                    ),
                                  ],
                                ),
                              ],
                              const SizedBox(height: 10),
                              Text(
                                s.description,
                                style: AppTextStyles.mono.copyWith(
                                  fontSize: 12,
                                  fontFamily:
                                      GoogleFonts.blinker().fontFamily,
                                ),
                              ),
                            ],
                          ),
                        ),

                        // Expand toggle
                        const SizedBox(width: 12),
                        GestureDetector(
                          onTap: () => setState(() => _expanded = !_expanded),
                          child: MouseRegion(
                            cursor: SystemMouseCursors.click,
                            child: AnimatedRotation(
                              duration: const Duration(milliseconds: 200),
                              turns: _expanded ? 0.5 : 0,
                              child: const Icon(
                                Icons.keyboard_arrow_down_rounded,
                                color: AppColors.textMuted,
                                size: 20,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),

              // ── Input / Output row ─────────────────────────────────────────
              Padding(
                padding: const EdgeInsets.fromLTRB(72, 18, 20, 18),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'INPUT',
                            style: AppTextStyles.monoLabel.copyWith(
                              fontSize: 9,
                            ),
                          ),
                          const SizedBox(height: 5),
                          Text(
                            s.input,
                            style: AppTextStyles.mono.copyWith(
                              color: s.accentColor,
                              fontSize: 10,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 24),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'OUTPUT',
                            style: AppTextStyles.monoLabel.copyWith(
                              fontSize: 9,
                            ),
                          ),
                          const SizedBox(height: 5),
                          Text(
                            s.output,
                            style: AppTextStyles.mono.copyWith(
                              color: AppColors.textPrimary,
                              fontSize: 10,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // ── Expandable detail bullets ──────────────────────────────────
              AnimatedCrossFade(
                duration: const Duration(milliseconds: 220),
                crossFadeState: _expanded
                    ? CrossFadeState.showFirst
                    : CrossFadeState.showSecond,
                firstChild: Container(
                  decoration: BoxDecoration(
                    color: AppColors.bg3,
                    border: Border(top: BorderSide(color: AppColors.border)),
                  ),
                  padding: const EdgeInsets.fromLTRB(72, 14, 20, 16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: s.bullets
                        .map(
                          (b) => Padding(
                            padding: const EdgeInsets.only(bottom: 7),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.center,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  '\u2022',
                                  style: AppTextStyles.monoSmall.copyWith(
                                    color: s.accentColor,
                                    fontSize: 22,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    b,
                                    style: AppTextStyles.monoSmall.copyWith(
                                      fontSize: 12,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        )
                        .toList(),
                  ),
                ),
                secondChild: const SizedBox.shrink(),
              ),
            ],
          ),
        ),
        // Colored left border overlay
        Positioned(
          left: 1.2,
          top: 1,
          bottom: 1,
          child: Container(
            width: 5,
            decoration: BoxDecoration(
              color: s.accentColor,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(8),
                bottomLeft: Radius.circular(8),
              ),
            ),
          ),
        ),
      ],
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
