import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../../core/routes/app_routes.dart';
import '../../widgets/custom_button.dart';
import '../../widgets/custom_textfield.dart';

class SeniorSetupScreen extends StatefulWidget {
  const SeniorSetupScreen({super.key});

  @override
  State<SeniorSetupScreen> createState() =>
      _SeniorSetupScreenState();
}

class _SeniorSetupScreenState
    extends State<SeniorSetupScreen> {

  final houseController = TextEditingController();
  final addressController = TextEditingController();
  final nameController = TextEditingController();
  final ageController = TextEditingController();
  final medicalController = TextEditingController();

  bool isLoading = false;

  // ⭐ Gender selection
  String selectedGender = "male";

  final String apiUrl =
      "http://172.20.10.8:5000/api/household/create";

  // =================================================
  Future<void> createHousehold() async {

    if (houseController.text.isEmpty ||
        nameController.text.isEmpty) {
      showMessage("Please fill required fields");
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
          "house_name": houseController.text.trim(),
          "address": addressController.text.trim(),
          "person_name": nameController.text.trim(),
          "age": int.tryParse(ageController.text) ?? 0,
          "gender": selectedGender,      // ⭐ IMPORTANT
          "medical_notes":
              medicalController.text.trim(),
        }),
      );

      debugPrint("STATUS: ${response.statusCode}");
      debugPrint("BODY: ${response.body}");

      final data = jsonDecode(response.body);

      if (!mounted) return;

      if (response.statusCode == 200) {

        final String inviteCode =
            data["invite_code"];

        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text("House Created"),
            content: Text(
              "Invite Code:\n\n$inviteCode\n\n"
              "Share this with caregivers.",
            ),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.pushReplacementNamed(
                    context,
                    AppRoutes.home,
                  );
                },
                child: const Text("OK"),
              )
            ],
          ),
        );

      } else {
        showMessage(
            data["error"] ?? "Creation failed");
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
      appBar: AppBar(
        title: const Text("Household Setup"),
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),

        child: Column(
          crossAxisAlignment:
              CrossAxisAlignment.stretch,

          children: [

            const Text(
              "Create Household",
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 20),

            CustomTextField(
              hint: "House Name",
              controller: houseController,
            ),

            const SizedBox(height: 12),

            CustomTextField(
              hint: "Address",
              controller: addressController,
            ),

            const SizedBox(height: 20),

            const Divider(),

            const SizedBox(height: 10),

            const Text(
              "Monitored Person Details",
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 12),

            CustomTextField(
              hint: "Full Name",
              controller: nameController,
            ),

            const SizedBox(height: 12),

            CustomTextField(
              hint: "Age",
              controller: ageController,
              keyboardType: TextInputType.number,
            ),

            const SizedBox(height: 12),

            // ⭐ GENDER DROPDOWN
            DropdownButtonFormField<String>(
              value: selectedGender,
              decoration: const InputDecoration(
                labelText: "Gender",
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(
                    value: "male",
                    child: Text("Male")),
                DropdownMenuItem(
                    value: "female",
                    child: Text("Female")),
                DropdownMenuItem(
                    value: "other",
                    child: Text("Other")),
              ],
              onChanged: (value) {
                setState(() {
                  selectedGender = value!;
                });
              },
            ),

            const SizedBox(height: 12),

            CustomTextField(
              hint: "Medical Notes",
              controller: medicalController,
            ),

            const SizedBox(height: 24),

            CustomButton(
              text: isLoading
                  ? "Creating..."
                  : "CREATE HOUSEHOLD",
              onPressed:
                  isLoading ? null : createHousehold,
            ),
          ],
        ),
      ),
    );
  }
}
