import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'app_colors.dart';

class AppTextStyles {
  AppTextStyles._();

  // ── Display / Headings  (Google Sans) ───────────────────────────────────────
  static TextStyle get displayLarge => GoogleFonts.blinker(
    fontSize: 28,
    fontWeight: FontWeight.w700,
    color: AppColors.textPrimary,
    letterSpacing: 1.5,
  );

  static TextStyle get displayMedium => GoogleFonts.blinker( //heading
    fontSize: 30,
    fontWeight: FontWeight.w700,
    color: AppColors.textPrimary,
    letterSpacing: 1.5
  );

  static TextStyle get heading => GoogleFonts.blinker( //pipleine headings
    fontSize: 15,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  static TextStyle get subheading => GoogleFonts.blinker(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  // ── KPI numbers (Google Sans bold) ───────────────────────────────────────────────
  static TextStyle kpiValue(Color color) => GoogleFonts.blinker(
    fontSize: 34,
    fontWeight: FontWeight.w700,
    color: color,
    letterSpacing: 1.5
  );

  static TextStyle kpiValueSmall(Color color) => GoogleFonts.blinker(
    fontSize: 20,
    fontWeight: FontWeight.w700,
    color: color,
    height: 1.0,
  );

  static TextStyle get kpiUnit => GoogleFonts.blinker(
    fontSize: 14,
    fontWeight: FontWeight.w600,
    color: AppColors.textMuted,
  );

  // ── Body / Mono  (JetBrains Mono) ────────────────────────────────────────
  static TextStyle get mono => GoogleFonts.blinker( //pipeline health
    fontSize: 12,
    fontWeight: FontWeight.w600,
    color: AppColors.textSecondary,
    letterSpacing: 1
  );

  static TextStyle get monoSmall => GoogleFonts.blinker( //graphs axis
    fontSize: 12,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 1,
  );

  static TextStyle get monoLabel => GoogleFonts.blinker( //section headers
    fontSize: 12,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 1,
  );

  static TextStyle monoColored(Color color) => GoogleFonts.blinker(
    fontSize: 11,
    fontWeight: FontWeight.w500,
    color: color,
  );

  // ── Nav labels ────────────────────────────────────────────────────────────
  static TextStyle get navLabel => GoogleFonts.blinker(
    fontSize: 13,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
  );

  static TextStyle get navLabelActive => GoogleFonts.blinker(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: AppColors.cyan,
  );

  static TextStyle get navSectionLabel => GoogleFonts.blinker(
    fontSize: 9,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary.withOpacity(0.5),
    letterSpacing: 0.2,
  );

  // ── Table ────────────────────────────────────────────────────────────────
  static TextStyle get tableHeader => GoogleFonts.blinker( //column headers
    fontSize: 12,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
    letterSpacing: 0.12,
  );

  static TextStyle get tableCell => GoogleFonts.blinker( //cell values
    fontSize: 11,
    fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
    letterSpacing: 1
  );

  static TextStyle tableCellColored(Color color) => GoogleFonts.blinker( //colored cell values
    fontSize: 11,
    fontWeight: FontWeight.w500,
    color: color,
    letterSpacing: 1
  );
}
