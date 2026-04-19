import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  // ── Backgrounds ───────────────────────────────────────────────────────────
  static const Color bg0 = Color(0xFF050810);
  static const Color bg1 = Color(0xFF0A0F1E);
  static const Color bg2 = Color(0xFF0F1628);
  static const Color bg3 = Color(0xFF161E35);

  // ── Borders ───────────────────────────────────────────────────────────────
  static const Color border = Color(0xFF1E2D4A);
  static const Color borderGlow = Color(0x3300C8FF);

  // ── Accent colours ────────────────────────────────────────────────────────
  static const Color cyan = Color(0xFF00C8FF);
  static const Color cyanDim = Color(0x6600C8FF);
  static const Color cyanFaint = Color(0x1800C8FF);

  static const Color amber = Color(0xFFF0A500);
  static const Color amberDim = Color(0x66F0A500);
  static const Color amberFaint = Color(0x18F0A500);

  static const Color green = Color(0xFF00E5A0);
  static const Color greenFaint = Color(0x1800E5A0);

  static const Color red = Color(0xFFFF4466);
  static const Color redFaint = Color(0x18FF4466);

  static const Color purple = Color(0xFFA78BFA);
  static const Color orange = Color(0xFFFB923C);

  // ── Text ──────────────────────────────────────────────────────────────────
  static const Color textPrimary = Color(0xFFE8F0FF);
  static const Color textSecondary = Color(0xFF6B82A8);
  static const Color textMuted = Color.fromARGB(255, 77, 90, 112);

  // ── Route chart colours (10 distinct hues) ────────────────────────────────
  static const List<Color> routeColors = [
    Color(0xFF00C8FF),
    Color(0xFFF0A500),
    Color(0xFF00E5A0),
    Color(0xFFFF6B9D),
    Color(0xFFA78BFA),
    Color(0xFFFB923C),
    Color(0xFF34D399),
    Color(0xFF60A5FA),
    Color(0xFFF472B6),
    Color(0xFFFACC15),
  ];
}
