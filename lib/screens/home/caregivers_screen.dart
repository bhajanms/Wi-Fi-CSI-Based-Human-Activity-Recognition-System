import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class CaregiversScreen extends StatefulWidget {
  const CaregiversScreen({super.key});

  @override
  State<CaregiversScreen> createState() => _CaregiversScreenState();
}

class _CaregiversScreenState extends State<CaregiversScreen> {
  List approvedUsers = [];
  List pendingRequests = [];
  bool isLoading = true;

  final String baseUrl = "http://172.20.10.8:5000/api/access";

  // =================================================
  // LOAD DATA FROM BACKEND
  // =================================================
  Future<void> loadData() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString("token");

    if (token == null) return;

    try {
      final approvedRes = await http.get(
        Uri.parse("$baseUrl/approved"),
        headers: {"Authorization": "Bearer $token"},
      );

      final pendingRes = await http.get(
        Uri.parse("$baseUrl/pending"),
        headers: {"Authorization": "Bearer $token"},
      );

      setState(() {
        approvedUsers = jsonDecode(approvedRes.body);
        pendingRequests = jsonDecode(pendingRes.body);
        isLoading = false;
      });
    } catch (e) {
      setState(() => isLoading = false);
    }
  }

  @override
  void initState() {
    super.initState();
    loadData();
  }

  // =================================================
  // APPROVE REQUEST
  // =================================================
  Future<void> approveRequest(int id) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString("token");

    await http.post(
      Uri.parse("$baseUrl/approve/$id"),
      headers: {"Authorization": "Bearer $token"},
    );

    loadData();
  }

  // =================================================
  // TOGGLE ACCESS
  // =================================================
  Future<void> toggleAccess(int id, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString("token");

    final response = await http.post(
      Uri.parse("$baseUrl/toggle/$id"),
      headers: {
        "Authorization": "Bearer $token",
        "Content-Type": "application/json",
      },
    body: jsonEncode({"enabled": value}),
  );

  debugPrint("TOGGLE STATUS: ${response.statusCode}");
  debugPrint("TOGGLE BODY: ${response.body}");

  loadData();
}

  // =================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Manage Caregivers")),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              children: [
                // =============================
                // PENDING REQUESTS
                // =============================
                if (pendingRequests.isNotEmpty)
                  const Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                      "Pending Requests",
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                  ),

                ...pendingRequests.map((req) => Card(
                      child: ListTile(
                        leading: const Icon(Icons.person_add),
                        title: Text(req["name"]),
                        subtitle: const Text("Waiting for approval"),
                        trailing: ElevatedButton(
                          onPressed: () => approveRequest(req["request_id"]),
                          child: const Text("Approve"),
                        ),
                      ),
                    )),

                // =============================
                // APPROVED USERS
                // =============================
                const Padding(
                  padding: EdgeInsets.all(12),
                  child: Text(
                    "Approved Caregivers",
                    style:
                        TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),

                ...approvedUsers.map((user) => Card(
                      child: ListTile(
                        leading: const Icon(Icons.person),
                        title: Text(user["name"]),
                        subtitle: Text(user["role"]),
                        trailing: Switch(
                          value: user["enabled"] ?? true,
                          onChanged: (value) =>
                              toggleAccess(user["id"], value),
                        ),
                      ),
                    )),
              ],
            ),
    );
  }
}
