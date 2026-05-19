import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AlertPreferencesScreen extends StatefulWidget {
  const AlertPreferencesScreen({super.key});

  @override
  State<AlertPreferencesScreen> createState() =>
      _AlertPreferencesScreenState();
}

class _AlertPreferencesScreenState
    extends State<AlertPreferencesScreen> {

  bool fallAlertEnabled = true;

  @override
  void initState() {
    super.initState();
    loadPreference();
  }

  // -----------------------------
  Future<void> loadPreference() async {
    final prefs = await SharedPreferences.getInstance();

    setState(() {
      fallAlertEnabled =
          prefs.getBool("fall_alert_enabled") ?? true;
    });
  }

  // -----------------------------
  Future<void> updatePreference(bool value) async {
    final prefs = await SharedPreferences.getInstance();

    await prefs.setBool("fall_alert_enabled", value);

    setState(() {
      fallAlertEnabled = value;
    });
  }

  // -----------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Alert Preferences"),
      ),
      body: ListView(
        children: [

          SwitchListTile(
            title: const Text("Fall Detection Alert"),
            subtitle: const Text(
              "Play alarm when a fall is detected",
            ),
            value: fallAlertEnabled,
            onChanged: (value) {
              updatePreference(value);
            },
          ),

          const Divider(),

          const Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              "If enabled, you will receive an alert sound when the monitored person falls.",
            ),
          ),
        ],
      ),
    );
  }
}