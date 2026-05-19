import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ActivityHistoryScreen extends StatefulWidget {
  const ActivityHistoryScreen({super.key});

  @override
  State<ActivityHistoryScreen> createState() =>
      _ActivityHistoryScreenState();
}

class _ActivityHistoryScreenState
    extends State<ActivityHistoryScreen> {

  List activities = [];
  Map<String, dynamic>? liveActivity;

  final String historyUrl =
      "http://172.20.10.8:5000/api/activity/history";

  final String latestUrl =
      "http://172.20.10.8:5000/api/activity/latest";

  bool loading = true;

  // ==============================
  // LOAD DATA
  // ==============================
  Future<void> loadActivities() async {

    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString("token");

    try {

      final historyRes = await http.get(
        Uri.parse(historyUrl),
        headers: {
          "Authorization": "Bearer $token"
        },
      );

      final latestRes = await http.get(
        Uri.parse(latestUrl),
      );

      if (historyRes.statusCode == 200) {
        activities = jsonDecode(historyRes.body);
      }

      if (latestRes.statusCode == 200) {
        liveActivity = jsonDecode(latestRes.body);
      }

    } catch (e) {
      debugPrint("Error loading activities");
    }

    setState(() {
      loading = false;
    });
  }

  @override
  void initState() {
    super.initState();
    loadActivities();
  }

  // ==============================
  // ACTIVITY COLOR
  // ==============================
  Color getActivityColor(String activity) {

    switch (activity.toLowerCase()) {

      case "fall":
        return const Color.fromARGB(255, 242, 76, 93);

      case "walk":
        return Colors.green.shade100;

      case "sit":
        return Colors.orange.shade100;

      case "stand":
        return Colors.blue.shade100;

      default:
        return Colors.grey.shade200;
    }
  }

  // ==============================
  // ACTIVITY TILE
  // ==============================
  Widget activityTile(Map activity) {

    String activityName =
        activity["activity"].toString();

    return Card(
      color: getActivityColor(activityName),

      margin: const EdgeInsets.symmetric(
        horizontal: 10,
        vertical: 5,
      ),

      child: ListTile(
        leading: const Icon(Icons.directions_walk),

        title: Text(
          activityName,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),

        subtitle: Text(
          activity["time"].toString(),
        ),

        trailing: Text(
          "${((activity["confidence"] ?? 0) * 100).toStringAsFixed(0)}%",
        ),
      ),
    );
  }

  // ==============================
  // UI
  // ==============================
  @override
  Widget build(BuildContext context) {

    if (loading) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(

      appBar: AppBar(
        title: const Text("Activity History"),
      ),

      body: activities.isEmpty && liveActivity == null
          ? Center(

              child: Column(
                mainAxisAlignment:
                    MainAxisAlignment.center,

                children: const [

                  Icon(
                    Icons.history,
                    size: 80,
                    color: Colors.grey,
                  ),

                  SizedBox(height: 20),

                  Text(
                    "No Activity History",
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey,
                    ),
                  ),

                  SizedBox(height: 8),

                  Text(
                    "Activities will appear here once detected.",
                    style:
                        TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            )

          : ListView(

              children: [

                if (liveActivity != null)

                  Card(
                    color: Colors.blue.shade50,
                    margin: const EdgeInsets.all(10),

                    child: ListTile(
                      leading:
                          const Icon(Icons.flash_on),

                      title: Text(
                        "Live: ${liveActivity!["activity"]}",
                      ),

                      subtitle: Text(
                        liveActivity!["time"]
                            .toString(),
                      ),
                    ),
                  ),

                ...activities
                    .map((a) => activityTile(a))
                    .toList(),
              ],
            ),
    );
  }
}