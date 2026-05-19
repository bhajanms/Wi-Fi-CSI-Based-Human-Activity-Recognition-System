import 'package:flutter/material.dart';

// =================================================
// AUTH SCREENS
// =================================================
import '../../screens/login/login_screen.dart';
import '../../screens/login/register_screen.dart';
import '../../screens/splash/splash_screen.dart';

// =================================================
// ONBOARDING SCREENS
// =================================================
import '../../screens/setup/senior_setup_screen.dart';
import '../../screens/setup/join_household_screen.dart';
import '../../screens/setup/permission_wait_screen.dart';

// =================================================
// HOME / DASHBOARD
// =================================================
import '../../screens/home/activity_screen.dart';
import '../../screens/home/settings_screen.dart';

// =================================================
// SETTINGS SUB SCREENS
// =================================================
import '../../screens/home/activity_history_screen.dart';
import '../../screens/home/caregivers_screen.dart';
import '../../screens/home/alert_preferences_screen.dart';
import '../../screens/home/system_status_screen.dart';
import '../../screens/home/language_screen.dart';

class AppRoutes {

  // =================================================
  // STARTUP
  // =================================================
  static const splash = '/';

  // =================================================
  // AUTHENTICATION
  // =================================================
  static const login = '/login';
  static const register = '/register';

  // =================================================
  // ONBOARDING (FIRST-TIME FLOW)
  // =================================================
  static const seniorSetup = '/senior-setup';
  static const joinHousehold = '/join-household';
  static const permissionWait = '/permission-wait';

  // =================================================
  // MAIN APP
  // =================================================
  static const home = '/home';
  static const settings = '/settings';

  // =================================================
  // SETTINGS SUB-PAGES
  // =================================================
  static const activityHistory = '/activity-history';
  static const caregivers = '/caregivers';
  static const alerts = '/alerts';
  static const systemStatus = '/system-status';
  static const language = '/language';

  // =================================================
  // ROUTE MAP
  // =================================================
  static final Map<String, WidgetBuilder> routes = {

    // -------------------------
    // Startup
    // -------------------------
    splash: (_) => const SplashScreen(),

    // -------------------------
    // Auth
    // -------------------------
    login: (_) => const LoginScreen(),
    register: (_) => const RegisterScreen(),

    // -------------------------
    // Onboarding
    // -------------------------
    seniorSetup: (_) => const SeniorSetupScreen(),
    joinHousehold: (_) => const JoinHouseholdScreen(),
    permissionWait: (_) => const PermissionWaitScreen(),

    // -------------------------
    // Main App
    // -------------------------
    home: (_) => const ActivityScreen(),
    settings: (_) => const SettingsScreen(),

    // -------------------------
    // Settings sub-pages
    // -------------------------
    activityHistory: (_) => const ActivityHistoryScreen(),
    caregivers: (_) => const CaregiversScreen(),
    alerts: (_) => const AlertPreferencesScreen(),
    systemStatus: (_) => const SystemStatusScreen(),
    language: (_) => const LanguageScreen(),
  };
}
