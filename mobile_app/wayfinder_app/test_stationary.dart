import 'dart:math' as math;
import 'lib/core/ekf_navigation.dart';

// Simulate what the app does.
void main() {
  final ekf = EKFNavigation(
    lat0: 12.9716,
    lon0: 77.5946,
    initHeading: 0.0,
    initSpeed: 10.0, // Give it an initial speed of 10 m/s (as if we just lost GPS)
  );
  
  double pdrLat = 12.9716;
  double pdrLon = 77.5946;
  
  bool _confirmedStationary = true;
  bool _alignDone = false;
  double _mlSpeedMs = 0.0;
  
  print('time_s,ekf_v,ekf_lat,ekf_lon,pdrLat,pdrLon,source');

  for (int i = 0; i <= 600; i++) {
    double dt = 0.1;
    double time_s = i * 0.1;

    // Simulate sensor noise that prevents stationary confirmation
    if (i > 100) { // after 10 seconds, imagine noise makes it think it's moving
       _confirmedStationary = false;
    }

    double validatedSpeed = 0.0;
    double speedVariance = 1.0;
    String source = "";

    if (_confirmedStationary) {
      ekf.updateZupt();
      _mlSpeedMs = 0.0;
      validatedSpeed = 0.0;
      speedVariance = 0.01;
      source = "ZUPT";
    } else {
      if (!_alignDone) {
        validatedSpeed = 0.0;
        speedVariance = 2.0;
        _mlSpeedMs = 0.0;
        source = "AI_GATE(v=0,var=2.0)";
      } else {
        // AI model
      }
    }

    // ALWAYS update speed before predict
    ekf.updateSpeed(validatedSpeed, variance: speedVariance);

    // Predict
    ekf.predict(dt, 0.0);
    ekf.updateNhc();

    NavState navState = ekf.state;
    double targetLat = navState.lat;
    double targetLon = navState.lon;

    // Map Matching (simulated as inactive if stationary or v < 0.5)
    bool mapMatched = false;
    if (!_confirmedStationary && _mlSpeedMs > 0.5) {
       // simulate map matching
    }

    // Marker EMA
    if ((targetLat - pdrLat).abs() > 0.005 || (targetLon - pdrLon).abs() > 0.005) {
      pdrLat = targetLat;
      pdrLon = targetLon;
    } else {
      pdrLat = pdrLat * 0.8 + targetLat * 0.2;
      pdrLon = pdrLon * 0.8 + targetLon * 0.2;
    }

    if (i % 10 == 0) {
      print('${time_s.toStringAsFixed(1)},${navState.speed.toStringAsFixed(4)},${navState.lat.toStringAsFixed(6)},${navState.lon.toStringAsFixed(6)},${pdrLat.toStringAsFixed(6)},${pdrLon.toStringAsFixed(6)},$source');
    }
  }
}
