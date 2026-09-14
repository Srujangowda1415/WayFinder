/// main.dart — WayFinder App
/// Calls nav.init() at startup for IP geolocation before showing any UI.

library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'core/navigation_service.dart';
import 'screens/home_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: Color(0xFF0A0D14),
  ));
  runApp(const WayFinderApp());
}

class WayFinderApp extends StatelessWidget {
  const WayFinderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) {
        final nav = NavigationService();
        nav.init(); // IP geolocation (no permissions)
        return nav;
      },
      child: MaterialApp(
        title: 'WayFinder',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          useMaterial3: true,
          scaffoldBackgroundColor: const Color(0xFF0A0D14),
          colorScheme: const ColorScheme.dark(
            primary: Color(0xFF4285F4),
            secondary: Color(0xFF34A853),
            surface: Color(0xFF1A1D27),
          ),
          fontFamily: 'Roboto',
        ),
        home: const HomeScreen(),
      ),
    );
  }
}
