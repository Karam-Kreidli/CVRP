import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/theme/app_text_styles.dart';
import 'package:flutter/material.dart';
class SectionLabel extends StatelessWidget {
  final String title;
  const SectionLabel({required this.title});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 16, height: 1, color: AppColors.textMuted),
        const SizedBox(width: 8),
        Text(
          title,
          style: AppTextStyles.monoLabel,
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(width: 8),
        Flexible(
          child: Container(height: 1, color: AppColors.textMuted),
        ),
      ],
    );
  }
}
