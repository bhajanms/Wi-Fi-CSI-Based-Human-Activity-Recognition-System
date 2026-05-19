class ApiService {
  // ⭐ Change for real device
  static const String baseUrl =
      "http://172.20.10.8:5000/api";

  // Default headers
  static Map<String, String> headers({
    String? token,
  }) {
    return {
      "Content-Type": "application/json",
      if (token != null) "Authorization": "Bearer $token",
    };
  }
}
