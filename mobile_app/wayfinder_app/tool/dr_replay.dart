/// dr_replay.dart — headless replay harness for the dead-reckoning path.
///
/// Runs the REAL `EKFNavigation` class that ships in the app (it is pure
/// Dart, no Flutter dependencies) and faithfully reproduces the DR branch of
/// `NavigationService._fuse()` — stationary detection, ZUPT, held speed,
/// complementary heading, predict, NHC and the EMA marker smoothing — against
/// recorded drive data.
///
/// This lets the dead-reckoning behaviour be tested without driving: feed it a
/// real drive, simulate a GNSS outage at a chosen offset, and compare the
/// dead-reckoned track against the recorded GPS truth.
///
/// Usage:
///   dart run tool/dr_replay.dart <replay.csv> <outage_start_s> <outage_len_s>
///
/// Input CSV columns:
///   time_s,ax,ay,az,gx,gy,gz,mx,my,mz,lat,lon,gps_speed_ms,gps_heading_deg
///
/// Output (stdout, CSV): per-tick truth vs dead-reckoned position and the
/// internal state, so it can be plotted.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../lib/core/ekf_navigation.dart';

/// Minimal OSM road index: drivable segments with nearest-segment snapping
/// and segment bearing. Mirrors what RoadMatcher does in the app.
class RoadIndex {
  static const drivable = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
    'residential', 'motorway_link', 'trunk_link', 'primary_link',
    'secondary_link', 'tertiary_link', 'living_street',
  };
  final List<double> ax = [], ay = [], bx = [], by = [];
  late double mLat, mLon;

  RoadIndex.fromOverpassJson(String jsonStr, double refLat) {
    mLat = 111320.0;
    mLon = 111320.0 * math.cos(refLat * math.pi / 180.0);
    final d = jsonDecode(jsonStr) as Map<String, dynamic>;
    for (final e in (d['elements'] as List)) {
      if (e['type'] != 'way' || e['geometry'] == null) continue;
      final tags = e['tags'] as Map<String, dynamic>?;
      if (tags == null || !drivable.contains(tags['highway'])) continue;
      final g = e['geometry'] as List;
      for (var i = 0; i + 1 < g.length; i++) {
        ax.add((g[i]['lon'] as num).toDouble() * mLon);
        ay.add((g[i]['lat'] as num).toDouble() * mLat);
        bx.add((g[i + 1]['lon'] as num).toDouble() * mLon);
        by.add((g[i + 1]['lat'] as num).toDouble() * mLat);
      }
    }
  }

  int get segmentCount => ax.length;

  /// Returns [distM, snapLat, snapLon, bearingRad] for the best segment.
  ///
  /// [headingRad], when given, gates candidates by bearing: a segment whose
  /// direction disagrees with where we believe we are travelling (in either
  /// sense, since ways are undirected) is rejected. Without this, greedy
  /// nearest-segment matching happily locks onto a crossing or parallel road
  /// at ~0 m and then confidently propagates along it — measured at one test
  /// window as a 0 m snap distance with 288 m of position error.
  List<double>? nearest(double lat, double lon,
      {double? headingRad, double maxBearingDiff = 0.79 /* ~45 deg */}) {
    final px = lon * mLon, py = lat * mLat;
    var best = double.infinity;
    var bi = -1;
    var bcx = 0.0, bcy = 0.0;
    for (var i = 0; i < ax.length; i++) {
      final vx = bx[i] - ax[i], vy = by[i] - ay[i];
      final l2 = vx * vx + vy * vy;
      if (l2 < 1e-9) continue;

      if (headingRad != null) {
        final segBrg = math.atan2(vx, vy);
        var diff = segBrg - headingRad;
        while (diff > math.pi) diff -= 2 * math.pi;
        while (diff < -math.pi) diff += 2 * math.pi;
        // undirected: fold to [0, pi/2] so both travel senses are allowed
        final folded = diff.abs() > math.pi / 2 ? math.pi - diff.abs() : diff.abs();
        if (folded > maxBearingDiff) continue;
      }

      var t = ((px - ax[i]) * vx + (py - ay[i]) * vy) / l2;
      t = t.clamp(0.0, 1.0);
      final cx = ax[i] + t * vx, cy = ay[i] + t * vy;
      final dd = (px - cx) * (px - cx) + (py - cy) * (py - cy);
      if (dd < best) {
        best = dd;
        bi = i;
        bcx = cx;
        bcy = cy;
      }
    }
    if (bi < 0) return null;
    final brg = math.atan2(bx[bi] - ax[bi], by[bi] - ay[bi]);
    return [math.sqrt(best), bcy / mLat, bcx / mLon, brg];
  }
}

/// Mirrors NavigationService's tilt-compensated compass (_updateCompass).
class Compass {
  double _compassRad = 0;
  double get rad => _compassRad;

  void update(double ax, double ay, double az, double mx, double my, double mz) {
    final aMag = math.sqrt(ax * ax + ay * ay + az * az);
    if (aMag < 0.5) return;
    final axN = ax / aMag, ayN = ay / aMag, azN = az / aMag;
    final pitch = math.asin(-axN.clamp(-1.0, 1.0));
    final roll = math.atan2(ayN, azN);
    final cp = math.cos(pitch), sp = math.sin(pitch);
    final cr = math.cos(roll), sr = math.sin(roll);
    final xH = mx * cp + my * sr * sp + mz * cr * sp;
    final yH = my * cr - mz * sr;
    double h = math.atan2(-yH, xH);
    if (h < 0) h += 2 * math.pi;
    double diff = h - _compassRad;
    while (diff > math.pi) diff -= 2 * math.pi;
    while (diff < -math.pi) diff += 2 * math.pi;
    _compassRad += 0.15 * diff;
    _compassRad %= (2 * math.pi);
    if (_compassRad < 0) _compassRad += 2 * math.pi;
  }
}

/// Mirrors NavigationService._updateMotionState() exactly.
class MotionDetector {
  final List<double> _mags = [];
  static const int bufSize = 20;
  static const int stationaryThresh = 10;
  static const int movingThresh = 3;
  int _statTicks = 0, _movTicks = 0;
  bool confirmedStationary = true;

  void pushAccel(double ax, double ay, double az) {
    _mags.add(math.sqrt(ax * ax + ay * ay + az * az));
    if (_mags.length > bufSize) _mags.removeAt(0);
  }

  void update(double gx, double gy, double gz) {
    if (_mags.length < bufSize) {
      confirmedStationary = true;
      return;
    }
    final minA = _mags.reduce(math.min);
    final maxA = _mags.reduce(math.max);
    final meanA = _mags.reduce((a, b) => a + b) / _mags.length;
    final gyroMag = math.sqrt(gx * gx + gy * gy + gz * gz);

    final raw = (maxA - minA) < 0.5 && meanA > 8.5 && meanA < 10.5 && gyroMag < 0.1;
    if (raw) {
      _movTicks = 0;
      _statTicks++;
      if (_statTicks >= stationaryThresh) confirmedStationary = true;
    } else {
      _statTicks = 0;
      _movTicks++;
      if (_movTicks >= movingThresh) confirmedStationary = false;
    }
  }
}

void main(List<String> args) {
  if (args.length < 3) {
    stderr.writeln('usage: dart run tool/dr_replay.dart <csv> <outage_start_s> <outage_len_s>');
    exit(2);
  }
  final path = args[0];
  final outageStart = double.parse(args[1]);
  final outageLen = double.parse(args[2]);
  final outageEnd = outageStart + outageLen;
  // Toggles so fixes can be A/B'd against current app behaviour.
  final fixYaw = (Platform.environment['FIX_YAW'] ?? '1') == '1';
  final fixHeadingSync = (Platform.environment['FIX_HDG'] ?? '1') == '1';
  stderr.writeln('[flags] fixYaw=$fixYaw fixHeadingSync=$fixHeadingSync');

  final lines = File(path).readAsLinesSync();
  final header = lines.first.split(',');
  int col(String n) => header.indexOf(n);
  final iT = col('time_s'), iAx = col('ax'), iAy = col('ay'), iAz = col('az');
  final iGx = col('gx'), iGy = col('gy'), iGz = col('gz');
  final iMx = col('mx'), iMy = col('my'), iMz = col('mz');
  final iLat = col('lat'), iLon = col('lon'), iSpd = col('gps_speed_ms');
  final iHdg = col('gps_heading_deg');

  final rows = <List<double>>[];
  for (var i = 1; i < lines.length; i++) {
    if (lines[i].trim().isEmpty) continue;
    rows.add(lines[i].split(',').map((e) => double.tryParse(e) ?? 0.0).toList());
  }

  // Initialise the EKF at the first sample, as the app does on first GPS fix.
  final lat0 = rows[0][iLat], lon0 = rows[0][iLon];
  final ekf = EKFNavigation(
    lat0: lat0,
    lon0: lon0,
    initHeading: rows[0][iHdg] * math.pi / 180.0,
    initSpeed: rows[0][iSpd],
  );

  // Optional OSM road aiding (OSM_JSON=<path>, disable by leaving unset).
  RoadIndex? roads;
  final osmPath = Platform.environment['OSM_JSON'];
  if (osmPath != null && osmPath.isNotEmpty && File(osmPath).existsSync()) {
    roads = RoadIndex.fromOverpassJson(
        File(osmPath).readAsStringSync(), rows[0][iLat]);
    stderr.writeln('[osm] ${roads.segmentCount} drivable segments loaded');
  }

  final compass = Compass();
  final motion = MotionDetector();

  double pdrLat = lat0, pdrLon = lon0;
  double drHoldSpeedMs = 0;
  bool inOutage = false, capturedThisOutage = false;

  stdout.writeln('time_s,truth_lat,truth_lon,dr_lat,dr_lon,gnss_on,'
      'ekf_speed,held_speed,truth_speed,stationary,heading_deg,err_m,compass_deg,yawrate,road_matched,snap_dist');

  double? outageStartLat, outageStartLon;
  double errAtOutageStart = 0;

  for (var i = 1; i < rows.length; i++) {
    final r = rows[i], rp = rows[i - 1];
    final t = r[iT];
    var dt = t - rp[iT];
    if (dt <= 0 || dt > 1.0) dt = 0.1;

    compass.update(r[iAx], r[iAy], r[iAz], r[iMx], r[iMy], r[iMz]);
    motion.pushAccel(r[iAx], r[iAy], r[iAz]);
    motion.update(r[iGx], r[iGy], r[iGz]);

    final gnssOn = !(t >= outageStart && t <= outageEnd);

    // Entry into DR: capture held speed once, exactly like _captureGnssAnchor().
    if (!gnssOn && !capturedThisOutage) {
      drHoldSpeedMs = ekf.speedMs.clamp(0.0, 55.0);
      capturedThisOutage = true;
      inOutage = true;
      outageStartLat = pdrLat;
      outageStartLon = pdrLon;
      errAtOutageStart = _distM(pdrLat, pdrLon, rp[iLat], rp[iLon]);
    }

    // Yaw rate about the VERTICAL axis, obtained by projecting the gyro
    // vector onto the measured gravity direction. Unlike the raw phone
    // Z-gyro this is independent of how the phone is mounted: gravity tells
    // us which way is down, and vehicle yaw is rotation about that axis.
    // FIX_YAW=0 falls back to the app's current raw-gz behaviour.
    double vehYawRate;
    if (fixYaw) {
      final gxN = r[iAx], gyN = r[iAy], gzN = r[iAz];
      final gN = math.sqrt(gxN * gxN + gyN * gyN + gzN * gzN);
      if (gN > 0.5) {
        // gravity vector points downward in the phone frame; rotation about
        // "down" is negative yaw in a N=0/clockwise convention.
        vehYawRate = -(r[iGx] * gxN + r[iGy] * gyN + r[iGz] * gzN) / gN;
      } else {
        vehYawRate = r[iGz];
      }
    } else {
      vehYawRate = r[iGz]; // raw phone Z gyro, as the app currently uses
    }

    var roadMatchedOut = false;
    var snapDistOut = -1.0;
    if (gnssOn) {
      // GNSS-active path: GPS anchors position and speed.
      ekf.updateGnss(r[iLat], r[iLon]);
      ekf.updateSpeed(r[iSpd], variance: 0.1);
      // Sync heading to GPS course-over-ground while genuinely moving. GPS
      // course is reliable above a few m/s and is the only absolute heading
      // reference available; without this the EKF heading is whatever the
      // gyro has drifted to by the time GNSS drops.
      if (fixHeadingSync && r[iSpd] > 3.0) {
        ekf.updateHeading(r[iHdg] * math.pi / 180.0, variance: 0.05);
      }
      if (motion.confirmedStationary) ekf.updateZupt();
      ekf.predict(dt, vehYawRate);
      ekf.updateNhc();
    } else {
      // ── DR path — mirrors NavigationService._fuse() exactly ──
      double validatedSpeed;
      double speedVariance;
      if (motion.confirmedStationary) {
        ekf.updateZupt();
        validatedSpeed = 0.0;
        speedVariance = 0.01;
      } else {
        validatedSpeed = drHoldSpeedMs;
        speedVariance = 1.0;
      }
      ekf.updateSpeed(validatedSpeed, variance: speedVariance);

      // ── OSM map aiding ────────────────────────────────────────────────
      // The road network is an independent, absolute reference. Its bearing
      // is a far better heading source than the phone gyro/compass, and
      // snapping removes cross-track error entirely.
      var roadMatched = false;
      if (roads != null && !motion.confirmedStationary) {
        final s0 = ekf.state;
        final hit = roads.nearest(s0.lat, s0.lon, headingRad: s0.heading);
        snapDistOut = hit == null ? -1.0 : hit[0];
        if (hit != null && hit[0] < 40.0) {
          roadMatched = true;
          roadMatchedOut = true;
          var brg = hit[3];
          // A road segment is undirected — pick the end that agrees with our
          // current heading, otherwise we would snap to oncoming traffic.
          double wrap(double a) {
            while (a > math.pi) a -= 2 * math.pi;
            while (a < -math.pi) a += 2 * math.pi;
            return a;
          }
          if (wrap(brg - s0.heading).abs() > math.pi / 2) {
            brg = brg + math.pi;
          }
          ekf.updateHeading(brg, variance: 0.02);
          // Snap laterally. The projection is perpendicular to the road, so
          // this corrects cross-track error without inventing along-track
          // progress — DR still supplies distance travelled.
          ekf.updateGnss(hit[1], hit[2]);
        }
      }

      // Compass complementary pull — but ONLY when the road has not already
      // given us a heading. The road bearing is ~1 deg accurate; the phone
      // compass in a car is tens of degrees off. Applying both means the
      // compass term immediately drags the heading back off the road, which
      // measurably made things worse (187 m vs 173 m) before this guard.
      double correctedYawRate = vehYawRate;
      if (!roadMatched) {
        double hErr = compass.rad - ekf.state.heading;
        while (hErr > math.pi) hErr -= 2 * math.pi;
        while (hErr < -math.pi) hErr += 2 * math.pi;
        correctedYawRate = vehYawRate + hErr * 0.5;
      }

      ekf.predict(dt, correctedYawRate);
      ekf.updateNhc();
    }

    // Marker update: safety gate + EMA, as in _fuse() steps 7-8.
    final s = ekf.state;
    const maxJumpDeg = 0.002;
    if ((s.lat - pdrLat).abs() > maxJumpDeg || (s.lon - pdrLon).abs() > maxJumpDeg) {
      pdrLat = s.lat;
      pdrLon = s.lon;
    } else {
      pdrLat = pdrLat * 0.8 + s.lat * 0.2;
      pdrLon = pdrLon * 0.8 + s.lon * 0.2;
    }

    final err = _distM(pdrLat, pdrLon, r[iLat], r[iLon]);
    stdout.writeln('${t.toStringAsFixed(2)},${r[iLat]},${r[iLon]},'
        '$pdrLat,$pdrLon,${gnssOn ? 1 : 0},'
        '${ekf.speedMs.toStringAsFixed(3)},${drHoldSpeedMs.toStringAsFixed(3)},'
        '${r[iSpd].toStringAsFixed(3)},${motion.confirmedStationary ? 1 : 0},'
        '${(ekf.state.heading * 180 / math.pi).toStringAsFixed(1)},'
        '${err.toStringAsFixed(2)},'
        '${(compass.rad * 180 / math.pi).toStringAsFixed(1)},'
        '${vehYawRate.toStringAsFixed(5)},'
        '${roadMatchedOut ? 1 : 0},${snapDistOut.toStringAsFixed(1)}');
  }

  if (inOutage && outageStartLat != null) {
    stderr.writeln('outage started at lat/lon $outageStartLat,$outageStartLon '
        '(pre-existing error ${errAtOutageStart.toStringAsFixed(1)} m)');
  }
}

double _distM(double la1, double lo1, double la2, double lo2) {
  const mPerDegLat = 111320.0;
  final mPerDegLon = mPerDegLat * math.cos(la1 * math.pi / 180.0);
  final dx = (lo2 - lo1) * mPerDegLon;
  final dy = (la2 - la1) * mPerDegLat;
  return math.sqrt(dx * dx + dy * dy);
}
