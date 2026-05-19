import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:lottie/lottie.dart';

import '../../core/routes/app_routes.dart';

class ActivityScreen extends StatefulWidget {
  const ActivityScreen({super.key});

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen> {

  String activity = "";
  double confidence = 0;
  String timestamp = "";

  bool isFall = false;
  bool staticAlert = false;

  bool isLoading = true;
  bool hasError = false;
  bool noData = false;
  bool accessDenied = false;

  bool fallAlertPlayed = false;
  bool staticAlertPlayed = false;

  final AudioPlayer player = AudioPlayer();

  final String apiUrl =
      "http://172.20.10.8:5000/api/activity/latest";

  // ---------------------------------------------
  // ACTIVITY ANIMATION
  // ---------------------------------------------
  Widget getActivityAnimation() {

    String act = activity.toLowerCase();
    String anim = "assets/animations/standing.json";

    if (act.isEmpty) {
      anim = "assets/animations/offline.json";
    }
    else if (act.contains("walk")) {
      anim = "assets/animations/walking.json";
    }
    else if (act.contains("sit")) {
      anim = "assets/animations/sitting.json";
    }
    else if (act.contains("lying")) {
      anim = "assets/animations/lying.json";
    }
    else if (act.contains("fall")) {
      anim = "assets/animations/fall.json";
    }

    return Lottie.asset(
      anim,
      width: 220,
      height: 220,
      repeat: true,
    );
  }

  // ---------------------------------------------
  // FETCH ACTIVITY FROM BACKEND
  // ---------------------------------------------
  Future<void> fetchActivity() async {

    try {

      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString("token");

      if (token == null) {
        setState(() {
          hasError = true;
          isLoading = false;
        });
        return;
      }

      final response = await http.get(
        Uri.parse(apiUrl),
        headers: {
          "Authorization": "Bearer $token",
        },
      );

      if (response.statusCode == 403) {
        setState(() {
          accessDenied = true;
          isLoading = false;
        });
        return;
      }

      if (response.statusCode == 200) {

        final data = jsonDecode(response.body);

        if (data == null ||
            data["activity"] == null ||
            data["activity"] == "") {

          setState(() {
            noData = true;
            isLoading = false;
          });

          return;
        }

        bool newFall = data["is_fall"] ?? false;
        bool newStaticAlert = data["static_alert"] ?? false;

        bool alertEnabled =
            prefs.getBool("fall_alert_enabled") ?? true;

        // FALL ALERT SOUND
        if (newFall && alertEnabled && !fallAlertPlayed) {
          player.play(AssetSource("sounds/alert.mp3"));
          fallAlertPlayed = true;
        }

        if (!newFall) {
          fallAlertPlayed = false;
        }

        // STATIC ALERT SOUND
        if (newStaticAlert && !staticAlertPlayed) {
          player.play(AssetSource("sounds/static_alert.mp3"));
          staticAlertPlayed = true;
        }

        if (!newStaticAlert) {
          staticAlertPlayed = false;
        }

        setState(() {
          activity = data["activity"];
          confidence = data["confidence"] ?? 0;
          timestamp = data["timestamp"] ?? "";

          isFall = newFall;
          staticAlert = newStaticAlert;

          isLoading = false;
          hasError = false;
          noData = false;
          accessDenied = false;
        });

      } else {
        setState(() {
          hasError = true;
          isLoading = false;
        });
      }

    } catch (e) {
      setState(() {
        hasError = true;
        isLoading = false;
      });
    }
  }

  // ---------------------------------------------
  // INIT
  // ---------------------------------------------
  @override
  void initState() {
    super.initState();

    fetchActivity();

    Future.doWhile(() async {
      await Future.delayed(
          const Duration(seconds: 3));
      await fetchActivity();
      return mounted;
    });
  }

  // ---------------------------------------------
  // BUILD CONTENT
  // ---------------------------------------------
  Widget buildContent() {

    if (isLoading) {
      return const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text("Loading activity...")
        ],
      );
    }

    if (accessDenied) {
      return const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.block, size: 80, color: Colors.red),
          SizedBox(height: 16),
          Text(
            "Access Denied",
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 8),
          Text(
            "Your monitoring permission was disabled by the senior.",
            textAlign: TextAlign.center,
          ),
        ],
      );
    }

    if (hasError) {
      return const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error, size: 80, color: Colors.red),
          SizedBox(height: 16),
          Text("Connection Error")
        ],
      );
    }

    if (noData) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [

          getActivityAnimation(),

          const SizedBox(height: 20),

          const Text(
            "Device Offline",
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 8),

          const Text(
            "No activity data from IoT device",
            style: TextStyle(color: Colors.grey),
          )
        ],
      );
    }

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [

        // ACTIVITY ANIMATION
        getActivityAnimation(),

        const SizedBox(height: 20),

        Text(
          activity,
          style: const TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
          ),
        ),

        const SizedBox(height: 10),

        Text(
          "Confidence: ${(confidence * 100).toStringAsFixed(1)}%",
        ),

        const SizedBox(height: 10),

        Text(
          "Last updated: $timestamp",
          style: const TextStyle(color: Colors.grey),
        ),

        const SizedBox(height: 20),

        // FALL ALERT
        if (isFall)
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.red.shade100,
            child: const Text(
              "⚠ FALL DETECTED!",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.red,
              ),
            ),
          ),

        // STATIC ACTIVITY ALERT
        if (staticAlert && !isFall)
          Container(
            margin: const EdgeInsets.only(top: 10),
            padding: const EdgeInsets.all(16),
            color: Colors.orange.shade100,
            child: const Text(
              "⚠ No movement detected for 2 minutes.\nPlease check the monitored person.",
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.orange,
              ),
            ),
          ),
      ],
    );
  }

  // ---------------------------------------------
  // MAIN SCREEN
  // ---------------------------------------------
  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Activity'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person),
            onPressed: () {
              Navigator.pushNamed(
                  context, AppRoutes.settings);
            },
          ),
        ],
      ),

      body: Center(
        child: buildContent(),
      ),
    );
  }
}