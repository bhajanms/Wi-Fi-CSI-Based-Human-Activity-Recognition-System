import 'dart:convert';
import 'package:http/http.dart' as http;

import 'api_service.dart';

class AuthService {
  // -------------------------
  // Register
  // -------------------------
  static Future<bool> register({
    required String name,
    required String email,
    required String phone,
    required String password,
    required String role,
  }) async {
    final url =
        Uri.parse("${ApiService.baseUrl}/auth/register");

    final response = await http.post(
      url,
      headers: ApiService.headers(),
      body: jsonEncode({
        "name": name,
        "email": email,
        "phone": phone,
        "password": password,
        "role": role,
      }),
    );

    return response.statusCode == 200;
  }

  // -------------------------
  // Login
  // -------------------------
  static Future<Map<String, dynamic>?> login({
    required String email,
    required String password,
  }) async {
    final url =
        Uri.parse("${ApiService.baseUrl}/auth/login");

    final response = await http.post(
      url,
      headers: ApiService.headers(),
      body: jsonEncode({
        "email": email,
        "password": password,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }

    return null;
  }

  // -------------------------
  // Get Profile
  // -------------------------
  static Future<Map<String, dynamic>?> getProfile(
      String token) async {
    final url =
        Uri.parse("${ApiService.baseUrl}/auth/me");

    final response = await http.get(
      url,
      headers: ApiService.headers(token: token),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }

    return null;
  }
}
