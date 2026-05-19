import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class SystemStatusScreen extends StatefulWidget {
  const SystemStatusScreen({super.key});

  @override
  State<SystemStatusScreen> createState() =>
      _SystemStatusScreenState();
}

class _SystemStatusScreenState
    extends State<SystemStatusScreen> {

  bool wifiConnected = false;
  bool sensorActive = false;
  bool isLoading = true;

  final String apiUrl =
      "http://172.20.10.8:5000/api/activity/latest";

  // ---------------------------------------------
  Future<void> checkSystemStatus() async {

    try {

      final prefs =
          await SharedPreferences.getInstance();
      final token = prefs.getString("token");

      final response = await http.get(
        Uri.parse(apiUrl),
        headers: {
          "Authorization": "Bearer $token",
        },
      );

      if (response.statusCode == 200) {

        final data = jsonDecode(response.body);

        bool hasActivity =
            data["activity"] != null &&
            data["activity"] != "";

        setState(() {
          wifiConnected = true;
          sensorActive = hasActivity;
          isLoading = false;
        });

      } else {

        setState(() {
          wifiConnected = false;
          sensorActive = false;
          isLoading = false;
        });

      }

    } catch (e) {

      setState(() {
        wifiConnected = false;
        sensorActive = false;
        isLoading = false;
      });

    }
  }

  // ---------------------------------------------
  @override
  void initState() {
    super.initState();
    checkSystemStatus();

    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 5));
      await checkSystemStatus();
      return mounted;
    });
    
  }

  // ---------------------------------------------
  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: const Text('System Status'),
      ),

      body: isLoading
          ? const Center(
              child: CircularProgressIndicator())
          : Column(
              children: [

                // WIFI STATUS
                ListTile(
                  leading: Icon(
                    wifiConnected
                        ? Icons.wifi
                        : Icons.wifi_off,
                    color: wifiConnected
                        ? Colors.green
                        : Colors.red,
                  ),
                  title: Text(
                    wifiConnected
                        ? 'WiFi Connected'
                        : 'WiFi Not Connected',
                  ),
                  subtitle: Text(
                    wifiConnected
                        ? 'Backend reachable'
                        : 'Cannot reach backend server',
                  ),
                ),

                const Divider(),

                // SENSOR STATUS
                ListTile(
                  leading: Icon(
                    sensorActive
                        ? Icons.sensors
                        : Icons.sensors_off,
                    color: sensorActive
                        ? Colors.green
                        : Colors.red,
                  ),
                  title: Text(
                    sensorActive
                        ? 'Sensors Active'
                        : 'Sensors Inactive',
                  ),
                  subtitle: Text(
                    sensorActive
                        ? 'Activity monitoring running'
                        : 'No sensor data available',
                  ),
                ),
              ],
            ),
    );
  }
}