import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../../widgets/custom_button.dart';
import '../../widgets/custom_textfield.dart';
import '../../core/routes/app_routes.dart';

class JoinHouseholdScreen extends StatefulWidget {
  const JoinHouseholdScreen({super.key});

  @override
  State<JoinHouseholdScreen> createState() =>
      _JoinHouseholdScreenState();
}

class _JoinHouseholdScreenState
    extends State<JoinHouseholdScreen> {

  final codeController = TextEditingController();

  bool isLoading = false;

  final String apiUrl =
      "http://172.20.10.8:5000/api/access/request";

  // =================================================
  // SEND REQUEST TO BACKEND
  // =================================================
  Future<void> requestAccess() async {

    if (codeController.text.isEmpty) {
      showMessage("Enter invite code");
      return;
    }

    setState(() => isLoading = true);

    try {
      final prefs =
          await SharedPreferences.getInstance();
      final token = prefs.getString("token");

      if (token == null) {
        showMessage("Not authenticated");
        return;
      }

      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer $token",
        },
        body: jsonEncode({
          "invite_code":
              codeController.text.trim(),
        }),
      );
      // ⭐ DEBUG OUTPUT
debugPrint("==== REQUEST DEBUG ====");
debugPrint("URL: $apiUrl");
debugPrint("STATUS: ${response.statusCode}");
debugPrint("BODY: ${response.body}");
debugPrint("=======================");

      final data = jsonDecode(response.body);

      if (!mounted) return;

      if (response.statusCode == 200) {

        showMessage("Request sent");

        Navigator.pushReplacementNamed(
          context,
          AppRoutes.permissionWait,
        );

      } else {
        showMessage(
            data["error"] ?? "Request failed");
      }

    } catch (e) {
      showMessage("Connection error");
    }

    if (mounted) {
      setState(() => isLoading = false);
    }
  }

  // =================================================
  void showMessage(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  // =================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar:
          AppBar(title: const Text("Join Household")),

      body: Padding(
        padding: const EdgeInsets.all(24),

        child: Column(
          mainAxisAlignment:
              MainAxisAlignment.center,

          children: [

            const Text(
              "Enter Invite Code",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 20),

            CustomTextField(
              hint: "HME-XXXXXX",
              controller: codeController,
            ),

            const SizedBox(height: 20),

            CustomButton(
              text: isLoading
                  ? "Sending..."
                  : "REQUEST ACCESS",
              onPressed:
                  isLoading ? null : requestAccess,
            ),
          ],
        ),
      ),
    );
  }
}
