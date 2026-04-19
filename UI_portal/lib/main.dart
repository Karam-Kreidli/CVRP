import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:RouteIQ_UI/utils/app_shell.dart';
import 'package:flutter/material.dart';
import 'package:flutter_web_plugins/url_strategy.dart';

void main() {
  usePathUrlStrategy();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'RouteIQ',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.bg0,
        colorScheme: const ColorScheme.dark(
          surface: AppColors.bg2,
          primary: AppColors.cyan,
          secondary: AppColors.amber,
        ),
        useMaterial3: true,
        // LinearProgressIndicator picks up ColorScheme.primary for value colour,
        // but each widget overrides it manually for precision.
      ),
      routerDelegate: shellRouter,
      routeInformationParser: shellRouteParser,
    );
  }
}
