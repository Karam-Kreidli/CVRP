import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
// ─── TAG ─────────────────────────────────────────────────────────────────────

class Tag extends StatelessWidget {
  final String label;
  final Color color;
  final bool fixWidth;
  const Tag({required this.label, required this.color, required this.fixWidth});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: fixWidth ? 100 : null,
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        border: Border.all(color: color.withOpacity(0.35)),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        label,
        style: GoogleFonts.blinker(
          fontSize: 10,
          fontWeight: FontWeight.w600,
          color: color,
          letterSpacing: 1
        ),
        overflow: TextOverflow.ellipsis,
        textAlign: TextAlign.center,
      ),
    );
  }
}
