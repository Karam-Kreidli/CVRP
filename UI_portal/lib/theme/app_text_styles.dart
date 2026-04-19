import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'app_colors.dart';

class AppTextStyles {
  AppTextStyles._();

  // ── Display / Headings  (Syne) ────────────────────────────────────────────
  static TextStyle get displayLarge => GoogleFonts.syne(
    fontSize: 28,
    fontWeight: FontWeight.w800,
    color: AppColors.textPrimary,
    letterSpacing: 0.2,
  );

  static TextStyle get displayMedium => GoogleFonts.syne(
    fontSize: 24,
    fontWeight: FontWeight.w800,
    color: AppColors.textPrimary,
  );

  static TextStyle get heading => GoogleFonts.syne(
    fontSize: 15,
    fontWeight: FontWeight.w700,
    color: AppColors.textPrimary,
  );

  static TextStyle get subheading => GoogleFonts.syne(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  // ── KPI numbers (Syne bold) ───────────────────────────────────────────────
  static TextStyle kpiValue(Color color) => GoogleFonts.syne(
    fontSize: 34,
    fontWeight: FontWeight.w800,
    color: color,
    height: 1.0,
  );

  static TextStyle kpiValueSmall(Color color) => GoogleFonts.syne(
    fontSize: 20,
    fontWeight: FontWeight.w700,
    color: color,
    height: 1.0,
  );

  static TextStyle get kpiUnit => GoogleFonts.syne(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    color: AppColors.textMuted,
  );

  // ── Body / Mono  (JetBrains Mono) ────────────────────────────────────────
  static TextStyle get mono => GoogleFonts.jetBrainsMono(
    fontSize: 11,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
  );

  static TextStyle get monoSmall => GoogleFonts.jetBrainsMono(
    fontSize: 9,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 0.1,
  );

  static TextStyle get monoLabel => GoogleFonts.jetBrainsMono(
    fontSize: 9,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 0.2,
  );

  static TextStyle monoColored(Color color) => GoogleFonts.jetBrainsMono(
    fontSize: 11,
    fontWeight: FontWeight.w500,
    color: color,
  );

  // ── Nav labels ────────────────────────────────────────────────────────────
  static TextStyle get navLabel => GoogleFonts.syne(
    fontSize: 13,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
  );

  static TextStyle get navLabelActive => GoogleFonts.syne(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: AppColors.cyan,
  );

  static TextStyle get navSectionLabel => GoogleFonts.jetBrainsMono(
    fontSize: 9,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 0.2,
  );

  // ── Table ────────────────────────────────────────────────────────────────
  static TextStyle get tableHeader => GoogleFonts.jetBrainsMono(
    fontSize: 9,
    fontWeight: FontWeight.w400,
    color: AppColors.textMuted,
    letterSpacing: 0.12,
  );

  static TextStyle get tableCell => GoogleFonts.jetBrainsMono(
    fontSize: 11,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
  );

  static TextStyle tableCellColored(Color color) => GoogleFonts.jetBrainsMono(
    fontSize: 11,
    fontWeight: FontWeight.w500,
    color: color,
  );
}
