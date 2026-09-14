/// road_matcher.dart — OpenStreetMap road-network aiding for dead reckoning.
///
/// WHY THIS EXISTS
/// During a GNSS outage the dominant error is NOT speed — it is heading.
/// Measured on a recorded drive replayed through the real EKF (see
/// tool/dr_replay.dart and PROTOTYPE_STATUS.md): held speed was within
/// ~1.2 m/s of truth, but heading drifted ~25 deg, and 428 m of travel at
/// 25 deg of heading error is ~174 m of cross-track error — essentially the
/// entire position error.
///
/// The OSM road network fixes precisely that, because a road's bearing is an
/// absolute heading reference. On the same drive, OSM road bearing matched
/// the true course to a median of 0.9 deg (p90 2.7 deg), versus tens of
/// degrees of error from the phone gyro/compass. Snapping to the road also
/// removes cross-track error outright, leaving only along-track error, which
/// held speed already handles well.
///
/// Measured effect on 30 s outages (real Dart EKF, recorded drive):
///   600 s window:  173 m -> 48 m      1200 s window: 340 m -> 5 m
/// It does not help everywhere: greedy nearest-segment matching can lock onto
/// a parallel road, which is why matching is bearing-gated and why the caller
/// should prefer an active route polyline when one exists (a single known
/// polyline has no wrong-road ambiguity).
///
/// DESIGN: roads are fetched from Overpass WHILE GNSS IS AVAILABLE and cached
/// in memory, so that matching works offline during the outage itself.
library;

import 'dart:convert';
import 'dart:math' as math;

import 'package:http/http.dart' as http;

class RoadMatch {
  final double distanceM;
  final double lat;
  final double lon;
  final double bearingRad;
  const RoadMatch(this.distanceM, this.lat, this.lon, this.bearingRad);
}

class RoadMatcher {
  static const Set<String> _drivable = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
    'residential', 'living_street', 'motorway_link', 'trunk_link',
    'primary_link', 'secondary_link', 'tertiary_link',
  };

  // Segment endpoints in local metres.
  final List<double> _ax = [], _ay = [], _bx = [], _by = [];
  double _mLat = 111320.0, _mLon = 111320.0;

  double? _loadedLat, _loadedLon;
  bool _loading = false;

  bool get isLoaded => _ax.isNotEmpty;
  int get segmentCount => _ax.length;

  /// True when the cache is missing or the vehicle has moved far enough from
  /// where it was built that a refresh is warranted.
  bool needsRefresh(double lat, double lon, {double thresholdM = 1500}) {
    if (_loadedLat == null) return true;
    final dx = (lon - _loadedLon!) * _mLon;
    final dy = (lat - _loadedLat!) * _mLat;
    return math.sqrt(dx * dx + dy * dy) > thresholdM;
  }

  /// Fetch drivable roads in a box around (lat, lon). Call this while GNSS is
  /// healthy; the result is what makes matching possible once it drops.
  Future<bool> loadAround(double lat, double lon, {double radiusM = 2500}) async {
    if (_loading) return false;
    _loading = true;
    try {
      final dLat = radiusM / 111320.0;
      final dLon = radiusM / (111320.0 * math.cos(lat * math.pi / 180.0));
      final s = lat - dLat, n = lat + dLat, w = lon - dLon, e = lon + dLon;
      final query = '[out:json][timeout:25];'
          'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
          'unclassified|residential|living_street|motorway_link|trunk_link|'
          'primary_link|secondary_link|tertiary_link)\$"]'
          '($s,$w,$n,$e);out geom;';

      final res = await http
          .post(
            Uri.parse('https://overpass-api.de/api/interpreter'),
            headers: {'User-Agent': 'WayFinder/1.0 (SIH prototype)'},
            body: {'data': query},
          )
          .timeout(const Duration(seconds: 30));
      if (res.statusCode != 200) return false;

      _parse(res.body, lat);
      _loadedLat = lat;
      _loadedLon = lon;
      return _ax.isNotEmpty;
    } catch (_) {
      // Offline or Overpass unavailable: matching simply stays disabled and
      // dead reckoning falls back to gyro/compass heading.
      return false;
    } finally {
      _loading = false;
    }
  }

  void _parse(String body, double refLat) {
    _ax.clear();
    _ay.clear();
    _bx.clear();
    _by.clear();
    _mLat = 111320.0;
    _mLon = 111320.0 * math.cos(refLat * math.pi / 180.0);

    final decoded = jsonDecode(body) as Map<String, dynamic>;
    final elements = decoded['elements'] as List?;
    if (elements == null) return;
    for (final el in elements) {
      if (el is! Map) continue;
      if (el['type'] != 'way' || el['geometry'] == null) continue;
      final tags = el['tags'];
      if (tags is! Map || !_drivable.contains(tags['highway'])) continue;
      final g = el['geometry'] as List;
      for (var i = 0; i + 1 < g.length; i++) {
        _ax.add((g[i]['lon'] as num).toDouble() * _mLon);
        _ay.add((g[i]['lat'] as num).toDouble() * _mLat);
        _bx.add((g[i + 1]['lon'] as num).toDouble() * _mLon);
        _by.add((g[i + 1]['lat'] as num).toDouble() * _mLat);
      }
    }
  }

  /// Nearest drivable segment, gated by bearing agreement with [headingRad].
  ///
  /// The bearing gate matters: without it, matching happily snaps to a
  /// crossing road at ~0 m and then propagates confidently along it. In replay
  /// testing that produced a 0 m snap distance alongside 288 m of position
  /// error. Ways are undirected, so agreement is folded to [0, pi/2].
  RoadMatch? match(double lat, double lon, double headingRad,
      {double maxDistanceM = 40.0, double maxBearingDiffRad = 0.79}) {
    if (_ax.isEmpty) return null;
    final px = lon * _mLon, py = lat * _mLat;
    var best = double.infinity;
    var bi = -1;
    var bcx = 0.0, bcy = 0.0;

    for (var i = 0; i < _ax.length; i++) {
      final vx = _bx[i] - _ax[i], vy = _by[i] - _ay[i];
      final l2 = vx * vx + vy * vy;
      if (l2 < 1e-9) continue;

      final segBrg = math.atan2(vx, vy);
      var diff = segBrg - headingRad;
      while (diff > math.pi) diff -= 2 * math.pi;
      while (diff < -math.pi) diff += 2 * math.pi;
      final folded = diff.abs() > math.pi / 2 ? math.pi - diff.abs() : diff.abs();
      if (folded > maxBearingDiffRad) continue;

      var t = ((px - _ax[i]) * vx + (py - _ay[i]) * vy) / l2;
      t = t.clamp(0.0, 1.0);
      final cx = _ax[i] + t * vx, cy = _ay[i] + t * vy;
      final dd = (px - cx) * (px - cx) + (py - cy) * (py - cy);
      if (dd < best) {
        best = dd;
        bi = i;
        bcx = cx;
        bcy = cy;
      }
    }
    if (bi < 0) return null;
    final dist = math.sqrt(best);
    if (dist > maxDistanceM) return null;

    // Orient the (undirected) segment to agree with our travel direction, so
    // we do not adopt the heading of oncoming traffic.
    var brg = math.atan2(_bx[bi] - _ax[bi], _by[bi] - _ay[bi]);
    var d = brg - headingRad;
    while (d > math.pi) d -= 2 * math.pi;
    while (d < -math.pi) d += 2 * math.pi;
    if (d.abs() > math.pi / 2) brg += math.pi;

    return RoadMatch(dist, bcy / _mLat, bcx / _mLon, brg);
  }
}
