import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../../core/routes/app_routes.dart';
import '../../widgets/custom_button.dart';
import '../../widgets/custom_textfield.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() =>
      _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {

  final emailController = TextEditingController();
  final passwordController =
      TextEditingController();

  bool isLoading = false; 

  // ⭐ Change if PC IP changes
  final String baseUrl =
      "http://172.20.10.8:5000/api/auth/login";

  // -------------------------------------------------
  Future<void> loginUser() async {

    if (emailController.text.isEmpty ||
        passwordController.text.isEmpty) {
      showError("Please enter email and password");
      return;
    }

    setState(() => isLoading = true);

    try {

      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {
          "Content-Type": "application/json"
        },
        body: jsonEncode({
          "email":
              emailController.text.trim(),
          "password":
              passwordController.text.trim(),
        }),
      );

      debugPrint("STATUS: ${response.statusCode}");
      debugPrint("BODY: ${response.body}");

      final data = jsonDecode(response.body);

      if (!mounted) return;

      if (response.statusCode == 200) {

        
        final String token = data["token"];

        final prefs =
            await SharedPreferences.getInstance();

        await prefs.setString("token", token);

        // Save role (optional but useful)
        final String role =
            (data["role"] ?? "")
                .toString()
                .toLowerCase();

        await prefs.setString("role", role);

        // =========================
        // Onboarding flags
        // =========================
        final bool setupComplete =
            data["setup_complete"] ?? false;

        final bool approved =
            data["approved"] ?? false;

        final bool requestPending =
            data["request_pending"] ?? false;

        // =========================
        // 👴 SENIOR FLOW
        // =========================
        if (role == "senior") {

          if (!setupComplete) {
            Navigator.pushReplacementNamed(
              context,
              AppRoutes.seniorSetup,
            );
          } else {
            Navigator.pushReplacementNamed(
              context,
              AppRoutes.home,
            );
          }

        }

        // =========================
        // 👩‍⚕️ CAREGIVER / MEMBER
        // =========================
        else {

          if (!approved && !requestPending) {
            Navigator.pushReplacementNamed(
              context,
              AppRoutes.joinHousehold,
            );
          }

          else if (!approved && requestPending) {
            Navigator.pushReplacementNamed(
              context,
              AppRoutes.permissionWait,
            );
          }

          else {
            Navigator.pushReplacementNamed(
              context,
              AppRoutes.home,
            );
          }

        }

      } else {
        showError(data["error"] ??
            "Invalid credentials");
      }

    } catch (e) {
      showError("Connection error");
    }

    if (mounted) {
      setState(() => isLoading = false);
    }
  }

  // -------------------------------------------------
  void showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  // -------------------------------------------------
  @override
  Widget build(BuildContext context) {

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding:
              const EdgeInsets.all(24),

          child: Column(
            mainAxisAlignment:
                MainAxisAlignment.center,

            children: [

              const Icon(
                Icons.monitor_heart,
                size: 80,
                color: Color.fromARGB(255, 40, 158, 0),
              ),

              const SizedBox(height: 16),

              const Text(
                'Smart Activity Monitoring',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight:
                      FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 30),

              CustomTextField(
                controller: emailController,
                hint: 'Email Address',
              ),

              const SizedBox(height: 15),

              CustomTextField(
                controller: passwordController,
                hint: 'Password',
                obscure: true,
              ),

              const SizedBox(height: 20),

              CustomButton(
                text: isLoading
                    ? 'Logging in...'
                    : 'LOGIN',
                onPressed:
                    isLoading ? null : loginUser,
              ),

              const SizedBox(height: 10),

              TextButton(
                onPressed: () {
                  Navigator.pushNamed(
                      context,
                      AppRoutes.register);
                },
                child: const Text(
                  'Register New Account',
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
