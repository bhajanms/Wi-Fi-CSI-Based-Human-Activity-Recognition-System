import 'package:flutter/material.dart';

import 'core/routes/app_routes.dart';
import 'core/theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Activity Monitoring',
      debugShowCheckedModeBanner: false,

      // ---------------------------
      // THEME
      // ---------------------------
      theme: AppTheme.lightTheme,

      // ---------------------------
      // START SCREEN
      // ---------------------------
      initialRoute: AppRoutes.splash, // ⭐ Start here

      // ---------------------------
      // ROUTES
      // ---------------------------
      routes: AppRoutes.routes,

      // ---------------------------
      // UNKNOWN ROUTE HANDLER
      // ---------------------------
      onUnknownRoute: (settings) {
        return MaterialPageRoute(
          builder: (_) => const Scaffold(
            body: Center(
              child: Text(
                'Page not found',
                style: TextStyle(fontSize: 18),
              ),
            ),
          ),
        );
      },
    );
  }
}
