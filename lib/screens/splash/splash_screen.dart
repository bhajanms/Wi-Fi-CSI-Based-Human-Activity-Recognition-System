import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import '../../core/routes/app_routes.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() =>
      _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {

  @override
  void initState() {
    super.initState();
    startupCheck();
  }

  void startupCheck() async {
    await Future.delayed(const Duration(seconds: 3));

    Navigator.pushReplacementNamed(
        context, AppRoutes.login);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [

            /// Popup Animation
            Lottie.asset(
              "assets/animations/splash_animation.json",
              width: 200,
              height: 200,
              repeat: true,
            ),

            const SizedBox(height: 20),

            const Text(
              "Smart Activity Monitoring",
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}