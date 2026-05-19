import 'dart:convert';
import 'package:http/http.dart' as http;

import 'api_service.dart';

class ActivityService {
  // -------------------------
  // Get Activity List
  // -------------------------
  static Future<List<dynamic>> getActivities(
      String token) async {
    final url =
        Uri.parse("${ApiService.baseUrl}/activity/list");

    final response = await http.get(
      url,
      headers: ApiService.headers(token: token),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }

    return [];
  }

  // -------------------------
  // Add Activity (Testing)
  // -------------------------
  static Future<bool> addActivity({
    required int personId,
    required String activity,
  }) async {
    final url =
        Uri.parse("${ApiService.baseUrl}/activity/add");

    final response = await http.post(
      url,
      headers: ApiService.headers(),
      body: jsonEncode({
        "person_id": personId,
        "activity": activity,
        "start_time": DateTime.now().toIso8601String(),
      }),
    );

    return response.statusCode == 200;
  }
}
