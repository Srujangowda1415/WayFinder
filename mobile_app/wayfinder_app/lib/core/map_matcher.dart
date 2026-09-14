import 'dart:math' as math;
import 'package:latlong2/latlong.dart';

class MapMatcher {
  /// Snaps a given point to the nearest line segment of the active route.
  /// Simulates the most likely emission state in a Viterbi algorithm where the 
  /// graph is constrained to a single pre-computed OSRM route.
  static LatLng? snapToRoute(LatLng currentPos, List<LatLng> routePoints, {double maxDistanceMeters = 50.0}) {
    if (routePoints.isEmpty || routePoints.length < 2) return null;

    double minDist = double.infinity;
    LatLng? bestSnap;

    for (int i = 0; i < routePoints.length - 1; i++) {
      final p1 = routePoints[i];
      final p2 = routePoints[i + 1];
      
      final snap = _projectPointToSegment(currentPos, p1, p2);
      
      // Calculate distance using simple equirectangular approximation for small distances
      final dist = _distanceMeters(currentPos, snap);
      
      if (dist < minDist) {
        minDist = dist;
        bestSnap = snap;
      }
    }

    if (minDist <= maxDistanceMeters) {
      return bestSnap;
    }
    
    return null; // Too far from route
  }

  static LatLng _projectPointToSegment(LatLng p, LatLng v, LatLng w) {
    // Convert to local cartesian (meters) relative to v
    final double latToM = 111320.0;
    final double lonToM = 111320.0 * math.cos(v.latitude * math.pi / 180.0);
    
    final px = (p.longitude - v.longitude) * lonToM;
    final py = (p.latitude - v.latitude) * latToM;
    
    final wx = (w.longitude - v.longitude) * lonToM;
    final wy = (w.latitude - v.latitude) * latToM;
    
    final l2 = wx * wx + wy * wy;
    if (l2 == 0.0) return v; // v == w
    
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line.
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    var t = (px * wx + py * wy) / l2;
    t = math.max(0.0, math.min(1.0, t)); // Constrain to segment
    
    final projLon = v.longitude + (wx * t) / lonToM;
    final projLat = v.latitude + (wy * t) / latToM;
    
    return LatLng(projLat, projLon);
  }

  static double _distanceMeters(LatLng p1, LatLng p2) {
    final double latToM = 111320.0;
    final double lonToM = 111320.0 * math.cos(p1.latitude * math.pi / 180.0);
    
    final dx = (p2.longitude - p1.longitude) * lonToM;
    final dy = (p2.latitude - p1.latitude) * latToM;
    
    return math.sqrt(dx * dx + dy * dy);
  }
}
