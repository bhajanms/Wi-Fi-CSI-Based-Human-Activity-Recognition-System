import 'package:flutter/material.dart';

class PermissionWaitScreen extends StatelessWidget {
  const PermissionWaitScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Icon(
                Icons.hourglass_top,
                size: 80,
                color: Colors.orange,
              ),

              SizedBox(height: 20),

              Text(
                "Waiting for Approval",
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),

              SizedBox(height: 10),

              Text(
                "Your request has been sent to the senior member.\n"
                "You will gain access once approved.",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
