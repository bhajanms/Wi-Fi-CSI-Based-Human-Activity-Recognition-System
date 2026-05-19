import 'package:flutter/material.dart';

class LanguageScreen extends StatelessWidget {
  const LanguageScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Language')),
      body: const Column(
        children: [
          RadioListTile(
            value: 'English',
            groupValue: 'English',
            onChanged: null,
            title: Text('English'),
          ),
          // RadioListTile(
          //   value: 'Malayalam',
          //   groupValue: 'English',
          //   onChanged: null,
          //   title: Text('Malayalam'),
          // ),
        ],
      ),
    );
  }
}
