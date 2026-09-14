/// navigation_service.dart
/// WayFinder — AI-Enhanced Dead Reckoning Navigation
///
/// Starting position  : IP geolocation (no permissions needed)
/// GNSS ON            : EKF fuses GNSS + IMU, PDR anchor = GPS position
/// GNSS OFF (DR mode) : speed held from GNSS-loss instant (+ZUPT) × compass heading propagates position
/// GNSS reacquired    : smooth drift correction via EKF

library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import 'package:path_provider/path_provider.dart';

import 'ekf_navigation.dart';
import 'map_matcher.dart';
import 'road_matcher.dart';
import 'speed_estimator.dart';

enum GnssStatus { active, denied, degraded }
enum NavMode { vehicle, walking }
enum InitStatus { loading, ipLocated, gpsActive, failed }
enum DrPhase { gnssNavigation, deadReckoning, gnssReacquired }

class TrackPoint {
  final double lat, lon;
  const TrackPoint(this.lat, this.lon);
}

class SearchResult {
  final String displayName;
  final double lat, lon;
  const SearchResult({required this.displayName, required this.lat, required this.lon});
}

class NavigationService extends ChangeNotifier {
  // ── Init ───────────────────────────────────────────────────────────────────
  InitStatus initStatus = InitStatus.loading;
  String     initCity   = '';

  // ── Navigation state ───────────────────────────────────────────────────────
  EKFNavigation?      _ekf;
  NavState?           currentState;
  DrPhase             drPhase = DrPhase.gnssNavigation;

  // Position (always up-to-date, regardless of source)
  double pdrLat = 12.9716;
  double pdrLon = 77.5946;

  // ── Last known GNSS state (DR anchor) ─────────────────────────────────────
  double _drAnchorLat = 12.9716;
  double _drAnchorLon = 77.5946;
  double _drAnchorHeadingRad = 0;

  // ── Speed estimation during GNSS outage ───────────────────────────────────
  // NOTE (2026-09-15): the CNN-GRU speed estimator is NO LONGER in the
  // navigation path. Measured on both the validation and test driver splits,
  // holding the last known GPS speed and zeroing it via ZUPT beat the model
  // on every metric — mean drift over moving 30s windows 23.7% vs 34.2%
  // (val) / 33.3% vs 42.3% (test), median position error 34m vs 100m (val)
  // / 43m vs 110m (test), and critically 3.5m vs 20m (val) / 31m vs 213m
  // (test) of false movement while parked. See PROTOTYPE_STATUS.md.
  // The model is left loaded but unused so it can be re-enabled if a future,
  // much larger training set makes it competitive (see DATA_COLLECTION_PLAN.md).
  final SpeedEstimatorModel _speedModel = SpeedEstimatorModel();
  double _mlSpeedMs = 0;       // speed currently driving DR (m/s) — also shown in UI
  double _mlVibrationScore = 0.0;
  double _drHoldSpeedMs = 0;   // speed captured at the moment GNSS was lost
  DateTime? _lastFuseTime;

  // ── OSM road aiding ───────────────────────────────────────────────────────
  // Roads are fetched while GNSS is healthy and cached, so that during an
  // outage the road bearing can supply an absolute heading reference. Heading
  // — not speed — is what actually breaks DR (see road_matcher.dart).
  final RoadMatcher _roads = RoadMatcher();
  bool _roadAidActive = false;      // matched a road on the last DR tick
  double _lastRoadSnapDistM = -1;
  bool get roadAidActive => _roadAidActive;
  int  get roadSegmentCount => _roads.segmentCount;

  // Alignment matrix (phone → vehicle frame), estimated online from GPS accel
  List<List<double>> _alignR = [[1,0,0],[0,1,0],[0,0,1]];
  bool _alignDone = false;
  final List<List<double>> _alignAccelBuf = [];  // raw accel samples (for alignment)
  final List<double>       _alignSpeedBuf = [];  // GPS speeds (for forward detection)

  // ── Motion Detection (always-running, independent of GNSS) ────────────────
  // Circular buffer of recent accel magnitudes at 10 Hz.
  // Used to determine STATIONARY vs MOVING using temporal consistency.
  final List<double> _motionAccelMags = [];
  static const int _motionBufSize = 20;   // 2 seconds at 10 Hz
  // Hysteresis counters — require N consecutive ticks to switch state
  int _stationaryTicks = 0;
  int _movingTicks     = 0;
  static const int _stationaryThresh = 10; // 1.0 s stationary before confirming
  static const int _movingThresh     = 3;  // 0.3 s moving before confirming
  bool _confirmedStationary = true; // starts stationary until proven otherwise
  // Expose for UI/debug
  bool get isConfirmedStationary => _confirmedStationary;

  // ── PDR (walking) ─────────────────────────────────────────────────────────
  double _pdrSpeedMs = 0;
  int    _stepCount  = 0;
  bool   _stepArmed  = false;
  double _prevMag    = 9.81;
  DateTime? _lastStepTs;

  // ── Compass ────────────────────────────────────────────────────────────────
  double _compassRad = 0;
  double? _lastCompassRad;
  double headingDeg  = 0;

  // ── IMU raw ────────────────────────────────────────────────────────────────
  double _ax = 0, _ay = 0, _az = 9.81;
  double _gx = 0, _gy = 0, _gz = 0;
  double _mx = 0, _my = 0, _mz = 0;
  double _vehYawRate = 0;
  bool   _hasMag = false;

  // ── GPS ────────────────────────────────────────────────────────────────────
  bool       _hasGps      = false;
  double     _gpsLat      = 0, _gpsLon = 0;
  double     _gpsSpeed    = 0;
  double     _gpsHeading  = 0;
  GnssStatus gnssStatus   = GnssStatus.denied;
  double     gnssAccuracyM = 999;
  bool       gnssSimDenied = false;
  DateTime?  _lastGpsTime;

  // ── Mode ───────────────────────────────────────────────────────────────────
  NavMode navMode = NavMode.walking;

  // ── Track ──────────────────────────────────────────────────────────────────
  final List<TrackPoint> trackPoints = [];
  double    _totalDistM = 0;
  DateTime? _startTime;

  // ── Metrics ────────────────────────────────────────────────────────────────
  double get speedKmh        => (navMode == NavMode.walking ? _pdrSpeedMs : _mlSpeedMs) * 3.6;
  double get vibrationScore  => _mlVibrationScore;
  double get totalDistanceKm => _totalDistM / 1000;
  int    get stepCount       => _stepCount;
  double get posUncertaintyM => _ekf?.positionUncertaintyM ?? 99;
  Duration get elapsed =>
      _startTime == null ? Duration.zero : DateTime.now().difference(_startTime!);
  bool   get isModelLoaded   => _speedModel.isLoaded;

  // ── Routing ────────────────────────────────────────────────────────────────
  List<LatLng> routePoints  = [];
  LatLng?      destLatLng;
  String       destName     = '';
  String       routeDist    = '';
  String       routeEta     = '';
  bool         routeLoading = false;

  // ── Running ────────────────────────────────────────────────────────────────
  bool _running = false;
  bool get isRunning => _running;

  // ── Data Recorder ─────────────────────────────────────────────────────────
  bool   _isRecording    = false;
  bool   get isRecording => _isRecording;
  final  List<String> _recordedRows = [];
  DateTime? _recordingStartTime;
  String    lastRecordingPath = '';

  /// Start a new CSV recording session.
  void startRecording() {
    if (_isRecording) return;
    _isRecording = true;
    _recordedRows.clear();
    _recordingStartTime = DateTime.now();
    // CSV header — units and coordinate conventions
    // ax/ay/az : m/s² (raw phone frame, sensors_plus AccelerometerEvent)
    // gx/gy/gz : rad/s (raw phone frame, sensors_plus GyroscopeEvent)
    // gps_lat/lon : degrees WGS-84
    // gps_speed : m/s (Geolocator Position.speed)
    // gps_heading : degrees true North (Geolocator Position.heading, 0–360)
    _recordedRows.add(
      'timestamp_iso8601,'
      'ax_ms2,ay_ms2,az_ms2,'
      'gx_rads,gy_rads,gz_rads,'
      'gps_lat_deg,gps_lon_deg,gps_speed_ms,gps_heading_deg'
    );
    notifyListeners();
  }

  /// Stop recording and write the CSV to external storage.
  Future<void> stopRecording() async {
    if (!_isRecording) return;
    _isRecording = false;
    notifyListeners();
    if (_recordedRows.length <= 1) return; // only header

    try {
      // Write to the app's external files dir so recordings are retrievable via
      //   adb pull /storage/emulated/0/Android/data/com.wayfinder.wayfinder_app/files/
      //
      // This previously hardcoded ".../com.wayfinder.app/files", which is a
      // DIFFERENT package than this app's actual id
      // (com.wayfinder.wayfinder_app, see android/app/build.gradle.kts). Under
      // scoped storage an app cannot write into another package's data dir, so
      // every recording failed into the catch below and was silently lost.
      // getExternalStorageDirectory() resolves the correct per-package path.
      final dir = await getExternalStorageDirectory()
          ?? await getApplicationDocumentsDirectory();
      if (!await dir.exists()) await dir.create(recursive: true);

      final ts   = _recordingStartTime?.millisecondsSinceEpoch ?? DateTime.now().millisecondsSinceEpoch;
      final file = File('${dir.path}/drive_$ts.csv');
      await file.writeAsString(_recordedRows.join('\n'));
      lastRecordingPath = file.path;
      print('[DATA_RECORDER] Saved ${_recordedRows.length - 1} rows to ${file.path}');
    } catch (e) {
      print('[DATA_RECORDER] Error saving: $e');
    }
  }

  StreamSubscription? _accelSub, _gyroSub, _magSub, _gpsSub;
  Timer? _fusionTimer;

  // ══════════════════════════════════════════════════════════════════════════
  // INIT
  // ══════════════════════════════════════════════════════════════════════════
  Future<void> init() async {
    initStatus = InitStatus.loading;
    notifyListeners();

    // NOTE: the CNN-GRU model is deliberately NOT loaded any more. It is no
    // longer in the navigation path (see the field declaration above), and its
    // weights were never declared under `assets:` in pubspec.yaml, so this
    // call always threw an unhandled async error at startup anyway.
    // To re-enable it you must BOTH restore this call AND add
    //   assets:
    //     - assets/model_weights.json
    //     - assets/norm_stats.json
    // to pubspec.yaml — otherwise rootBundle.loadString will keep failing.

    // IP Geolocation
    double lat = 12.9716, lon = 77.5946;
    String city = 'Bengaluru';
    try {
      final res = await http
          .get(Uri.parse('http://ip-api.com/json?fields=lat,lon,city'))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        final j = jsonDecode(res.body);
        if (j['lat'] != null && j['lon'] != null) {
          lat  = (j['lat'] as num).toDouble();
          lon  = (j['lon'] as num).toDouble();
          city = j['city'] as String? ?? 'Unknown';
          initStatus = InitStatus.ipLocated;
        }
      }
    } catch (_) {
      initStatus = InitStatus.failed;
    }
    if (initStatus == InitStatus.loading) initStatus = InitStatus.ipLocated;

    pdrLat = _drAnchorLat = lat;
    pdrLon = _drAnchorLon = lon;
    initCity = city;
    _initEKF(lat, lon, 0, 0);
    notifyListeners();
  }

  void _initEKF(double lat, double lon, double hRad, double speed) {
    _ekf = EKFNavigation(lat0: lat, lon0: lon, initHeading: hRad, initSpeed: speed);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // START / STOP
  // ══════════════════════════════════════════════════════════════════════════
  Future<void> start() async {
    if (_running) return;
    _running    = true;
    _startTime  = DateTime.now();
    _stepCount  = 0;
    _totalDistM = 0;
    trackPoints.clear();

    // Reset all navigation state to prevent stale data from previous session
    _mlSpeedMs = 0;
    _mlVibrationScore = 0;
    _drHoldSpeedMs = 0;
    _lastFuseTime = null;
    _motionAccelMags.clear();
    _stationaryTicks = 0;
    _movingTicks = 0;
    _confirmedStationary = true; // safe default: assume stationary until IMU says otherwise

    _accelSub = accelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 100),
    ).listen(_onAccel);

    _gyroSub = gyroscopeEventStream(
      samplingPeriod: const Duration(milliseconds: 100),
    ).listen((e) { _gx = e.x; _gy = e.y; _gz = e.z; });

    _magSub = magnetometerEventStream(
      samplingPeriod: const Duration(milliseconds: 100),
    ).listen(_onMag);

    // GPS — optional, graceful degradation
    try {
      LocationPermission perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.whileInUse || perm == LocationPermission.always) {
        _gpsSub = Geolocator.getPositionStream(
          locationSettings: AndroidSettings(
            accuracy: LocationAccuracy.best,
            intervalDuration: const Duration(seconds: 1),
            distanceFilter: 0,
          ),
        ).listen(_onGps, onError: (_) => _markGnssLost());
      }
    } catch (_) {}

    _fusionTimer = Timer.periodic(const Duration(milliseconds: 100), _fuse);
    notifyListeners();
  }

  void stop() {
    _running = false;
    _accelSub?.cancel(); _gyroSub?.cancel();
    _magSub?.cancel();   _gpsSub?.cancel();
    _fusionTimer?.cancel();
    // Reset motion state so next session starts clean
    _motionAccelMags.clear();
    _stationaryTicks = 0;
    _movingTicks = 0;
    _confirmedStationary = true;
    _mlSpeedMs = 0;
    _drHoldSpeedMs = 0;
    _lastFuseTime = null;
    notifyListeners();
  }

  void toggleGnssDenied() {
    gnssSimDenied = !gnssSimDenied;
    if (gnssSimDenied) _captureGnssAnchor();
    notifyListeners();
  }

  void setNavMode(NavMode m) { navMode = m; notifyListeners(); }

  // ══════════════════════════════════════════════════════════════════════════
  // IMU HANDLERS
  // ══════════════════════════════════════════════════════════════════════════
  void _onAccel(AccelerometerEvent e) {
    _ax = e.x; _ay = e.y; _az = e.z;
    _updateCompass();

    // Collect alignment data while GNSS is valid (for alignment matrix estimation)
    if (_hasGps && !gnssSimDenied && gnssStatus == GnssStatus.active) {
      _alignAccelBuf.add([_ax, _ay, _az]);
      _alignSpeedBuf.add(_gpsLat);
      if (_alignAccelBuf.length > 500) _alignAccelBuf.removeAt(0);
    }

    // Always update motion detection buffer (works in GNSS and DR mode)
    final aMag = math.sqrt(_ax*_ax + _ay*_ay + _az*_az);
    _motionAccelMags.add(aMag);
    if (_motionAccelMags.length > _motionBufSize) _motionAccelMags.removeAt(0);

    // Yaw rate for the EKF heading propagation. MUST be updated on every
    // accelerometer event, unconditionally.
    //
    // This was previously nested inside `if (_speedModel.isLoaded)`, together
    // with the (now removed) ML sample push. The model's weights were never
    // declared as assets in pubspec.yaml, so it never loaded, so this line
    // never ran and _vehYawRate stayed 0 for the entire session — meaning the
    // EKF's gyro heading propagation was dead and heading came only from the
    // compass complementary term. See PROTOTYPE_STATUS.md.
    _vehYawRate = _gz; // raw phone Z gyro for yaw rate

    if (navMode == NavMode.walking && _running) _detectStep();
  }

  /// Compute whether the vehicle is stationary.
  /// Uses temporal consistency over _motionBufSize samples to avoid
  /// flickering between STATIONARY and MOVING due to sensor noise.
  /// Also considers gyro magnitude to catch rotations without translation.
  void _updateMotionState() {
    if (_motionAccelMags.length < _motionBufSize) {
      // Not enough data yet — stay stationary (safe default)
      _confirmedStationary = true;
      return;
    }

    final minA = _motionAccelMags.reduce(math.min);
    final maxA = _motionAccelMags.reduce(math.max);
    final meanA = _motionAccelMags.reduce((a, b) => a + b) / _motionAccelMags.length;
    final gyroMag = math.sqrt(_gx*_gx + _gy*_gy + _gz*_gz);

    // Stationary criteria:
    // 1. Accel variance is small (< 0.5 m/s² range over 2s)
    // 2. Mean accel is near gravity (8.5–10.5 m/s²), meaning phone is still
    // 3. Gyro magnitude is small (< 0.1 rad/s), meaning phone isn't rotating
    final accelVarianceSmall = (maxA - minA) < 0.5;
    final nearGravity = meanA > 8.5 && meanA < 10.5;
    final gyroSmall = gyroMag < 0.1;

    final rawStationary = accelVarianceSmall && nearGravity && gyroSmall;

    if (rawStationary) {
      _movingTicks = 0;
      _stationaryTicks++;
      if (_stationaryTicks >= _stationaryThresh) {
        _confirmedStationary = true;
      }
    } else {
      _stationaryTicks = 0;
      _movingTicks++;
      if (_movingTicks >= _movingThresh) {
        _confirmedStationary = false;
      }
    }
  }

  void _onMag(MagnetometerEvent e) {
    _mx = e.x; _my = e.y; _mz = e.z;
    _hasMag = true;
    _updateCompass();
  }

  /// Tilt-compensated compass (Accel + Mag → azimuth)
  void _updateCompass() {
    if (!_hasMag) return;
    final aMag = math.sqrt(_ax*_ax + _ay*_ay + _az*_az);
    if (aMag < 0.5) return;

    final axN = _ax / aMag, ayN = _ay / aMag, azN = _az / aMag;
    final pitch = math.asin(-axN.clamp(-1.0, 1.0));
    final roll  = math.atan2(ayN, azN);
    final cp = math.cos(pitch), sp = math.sin(pitch);
    final cr = math.cos(roll),  sr = math.sin(roll);

    final xH = _mx * cp + _my * sr * sp + _mz * cr * sp;
    final yH = _my * cr - _mz * sr;

    double h = math.atan2(-yH, xH);
    if (h < 0) h += 2 * math.pi;

    double diff = h - _compassRad;
    while (diff >  math.pi) diff -= 2 * math.pi;
    while (diff < -math.pi) diff += 2 * math.pi;
    _compassRad += 0.15 * diff;
    _compassRad %= (2 * math.pi); // clamp to 0..2pi
    if (_compassRad < 0) _compassRad += 2 * math.pi;
    headingDeg = _compassRad * 180 / math.pi;
  }

  // ── Step detection (PDR walking) ──────────────────────────────────────────
  void _detectStep() {
    final mag = math.sqrt(_ax*_ax + _ay*_ay + _az*_az);
    final delta = (mag - _prevMag).abs();
    _prevMag = _prevMag * 0.8 + mag * 0.2;

    final now = DateTime.now();
    final sinceLastStep = _lastStepTs == null
        ? 9999
        : now.difference(_lastStepTs!).inMilliseconds;

    if (delta < 0.6) {
      _stepArmed = true;
    } else if (delta > 2.2 && _stepArmed && sinceLastStep > 280) {
      _stepCount++;
      _lastStepTs = now;
      _stepArmed  = false;
      _pdrSpeedMs = 0.75 / (sinceLastStep / 1000.0);

      const mPerDegLat = 111320.0;
      final mPerDegLon = mPerDegLat * math.cos(pdrLat * math.pi / 180);
      pdrLat += (0.75 * math.cos(_compassRad)) / mPerDegLat;
      pdrLon += (0.75 * math.sin(_compassRad)) / mPerDegLon;
    }

    if (_lastStepTs != null && now.difference(_lastStepTs!).inMilliseconds > 1500) {
      _pdrSpeedMs = 0;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // GPS HANDLER — GNSS state machine
  // ══════════════════════════════════════════════════════════════════════════
  void _onGps(Position p) {
    final wasActive = gnssStatus == GnssStatus.active && !gnssSimDenied;
    _hasGps      = true;
    _lastGpsTime = DateTime.now();
    gnssAccuracyM = p.accuracy;
    gnssStatus    = p.accuracy < 25 ? GnssStatus.active : GnssStatus.degraded;
    if (gnssStatus == GnssStatus.active) initStatus = InitStatus.gpsActive;

    _gpsLat     = p.latitude;
    _gpsLon     = p.longitude;
    _gpsSpeed   = p.speed;
    _gpsHeading = p.heading;

    if (!gnssSimDenied && gnssStatus == GnssStatus.active) {
      // ── GNSS active: anchor PDR and EKF to GPS truth ─────────────────────
      if (_ekf == null) {
        _initEKF(p.latitude, p.longitude, p.heading * math.pi / 180, p.speed);
      } else {
        _ekf!.updateGnss(p.latitude, p.longitude);
        _ekf!.updateSpeed(p.speed, variance: 0.1);
      }
      
      // Let pdrLat/pdrLon gracefully catch up to EKF state via EMA in _fuse
      // instead of violently snapping them here.

      // Save as DR anchor (used when GPS drops)
      _drAnchorLat = p.latitude;
      _drAnchorLon = p.longitude;
      _drAnchorHeadingRad = _compassRad;

      drPhase = DrPhase.gnssNavigation;

      // Compute alignment matrix once we have enough data
      if (!_alignDone && _alignAccelBuf.length >= 200) {
        _estimateAlignment(p.speed);
      }

      // Prefetch the surrounding road network WHILE GNSS is healthy, so that
      // road-based heading aiding still works once the signal drops (the
      // outage is exactly when we cannot download anything). Refreshes only
      // after the vehicle has moved well beyond the cached box.
      if (_roads.needsRefresh(p.latitude, p.longitude)) {
        _roads.loadAround(p.latitude, p.longitude).then((ok) {
          if (ok) notifyListeners();
        });
      }
    }

    // Reacquisition after DR: correct EKF drift
    if (!wasActive && gnssStatus == GnssStatus.active && !gnssSimDenied &&
        drPhase == DrPhase.deadReckoning) {
      _ekf?.updateGnss(p.latitude, p.longitude);
      drPhase = DrPhase.gnssReacquired;
    }

    notifyListeners();
  }

  void _markGnssLost() {
    gnssStatus = GnssStatus.denied;
    _captureGnssAnchor();
    notifyListeners();
  }

  void _captureGnssAnchor() {
    // Freeze DR anchor at the current reliable position
    _drAnchorLat = pdrLat;
    _drAnchorLon = pdrLon;
    _drAnchorHeadingRad = _compassRad;

    // Capture the speed we were travelling at the instant GNSS was lost.
    // This is the seed for dead reckoning: a 30s outage is a bounded
    // velocity-propagation problem from a KNOWN initial speed, not an
    // absolute-speed-inference problem. Prefer the EKF's filtered speed
    // over the raw GPS sample (less noise); fall back to raw GPS.
    final seed = _ekf?.speedMs ?? _gpsSpeed;
    _drHoldSpeedMs = seed.isFinite ? seed.clamp(0.0, 55.0) : 0.0;

    drPhase = DrPhase.deadReckoning;
  }

  // ── Simple online alignment estimation ───────────────────────────────────
  void _estimateAlignment(double gpsSpeed) {
    if (_alignAccelBuf.isEmpty) return;

    // Down vector = mean gravity
    final n = _alignAccelBuf.length;
    final down = [
      _alignAccelBuf.map((a) => a[0]).reduce((a, b) => a + b) / n,
      _alignAccelBuf.map((a) => a[1]).reduce((a, b) => a + b) / n,
      _alignAccelBuf.map((a) => a[2]).reduce((a, b) => a + b) / n,
    ];
    final dMag = math.sqrt(down[0]*down[0] + down[1]*down[1] + down[2]*down[2]);
    final zVec = down.map((d) => d / dMag).toList();

    // Forward vector ≈ [0, 1, 0] in phone portrait — use compass heading to rotate
    // Simple heuristic: x_vehicle is the horizontal direction of travel
    final ch = math.cos(_compassRad), sh = math.sin(_compassRad);
    var xVec = [sh, ch, 0.0];  // North-aligned horizontal forward

    // Orthogonalise xVec against zVec
    final dot = xVec[0]*zVec[0] + xVec[1]*zVec[1] + xVec[2]*zVec[2];
    xVec = [xVec[0] - dot*zVec[0], xVec[1] - dot*zVec[1], xVec[2] - dot*zVec[2]];
    final xMag = math.sqrt(xVec[0]*xVec[0] + xVec[1]*xVec[1] + xVec[2]*xVec[2]);
    if (xMag < 1e-6) return;
    xVec = xVec.map((v) => v / xMag).toList();

    // y = z × x
    final yVec = [
      zVec[1]*xVec[2] - zVec[2]*xVec[1],
      zVec[2]*xVec[0] - zVec[0]*xVec[2],
      zVec[0]*xVec[1] - zVec[1]*xVec[0],
    ];

    _alignR = [xVec, yVec, zVec];
    _alignDone = true;
  }

  List<double> _rotateVec(List<double> v, List<List<double>> R) =>
      [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
      ];

  // ══════════════════════════════════════════════════════════════════════════
  // FUSION LOOP (10 Hz)
  // ══════════════════════════════════════════════════════════════════════════
  void _fuse(Timer _) {
    if (_ekf == null) return;

    // ── dt calculation: clamp to max 0.15s to prevent velocity jumps ────────
    final now = DateTime.now();
    final dt = (_lastFuseTime != null)
        ? now.difference(_lastFuseTime!).inMilliseconds.clamp(50, 150) / 1000.0
        : 0.1;
    _lastFuseTime = now;

    // Detect GPS timeout → switch to DR
    if (_hasGps && _lastGpsTime != null) {
      final age = DateTime.now().difference(_lastGpsTime!).inSeconds;
      if (age > 3 && gnssStatus == GnssStatus.active && !gnssSimDenied) {
        _markGnssLost();
      }
    }

    final gnssOn = _hasGps && !gnssSimDenied && gnssStatus == GnssStatus.active;

    // Catch-all entry into dead reckoning. The timeout check above only fires
    // when gnssStatus is *active*, so a fix that DEGRADES (accuracy >= 25m)
    // rather than disappearing would drop us into the DR branch below without
    // ever capturing the speed DR now depends on — leaving the marker frozen
    // at 0, or worse, running on a hold speed from a previous outage.
    // Capturing here covers every route into DR, and is self-limiting because
    // _captureGnssAnchor() sets drPhase = deadReckoning.
    if (!gnssOn && drPhase != DrPhase.deadReckoning) {
      _captureGnssAnchor();
    }

    // ── Step 1: Always update motion state from raw IMU ─────────────────────
    // This uses _motionAccelMags which is filled in _onAccel regardless of GNSS.
    _updateMotionState();
    
    double validatedSpeed = 0.0;
    double speedVariance = 1.0;

    if (navMode == NavMode.vehicle) {
      // ── VEHICLE MODE ─────────────────────────────────────────────────────
      if (gnssOn) {
        // ── GNSS active path ─────────────────────────────────────────────
        // GPS speed is the ground truth — only predict heading rotation.
        // Speed measurement update is already done in _onGps().
        // Apply ZUPT when confirmed stationary to zero EKF velocity bias.
        if (_confirmedStationary) {
          _ekf!.updateZupt();
          _mlSpeedMs = 0;
          // Only predict heading (zero velocity predict):
          _ekf!.predict(dt, _vehYawRate);
        } else {
          // Keep the displayed speed live while GNSS is driving the filter.
          // Previously _mlSpeedMs was only ever written in the DR path (and
          // zeroed when stationary), so the on-screen speedometer read 0 the
          // whole time the vehicle was navigating normally on GPS.
          _mlSpeedMs = _ekf!.speedMs;
          _ekf!.predict(dt, _vehYawRate);
        }
        _ekf!.updateNhc();

      } else {
        // ── DR (GNSS-denied) path ────────────────────────────────────────
        //
        // ARCHITECTURE:
        //   1. Determine validated speed FIRST (held GNSS-loss speed, or 0
        //      if the IMU-based stationary detector says we're stopped)
        //   2. Apply ZUPT if stationary (corrects EKF internal velocity to 0)
        //   3. THEN call predict() using the validated speed in EKF state
        //   4. Position only changes if the validated speed is non-zero
        //
        // This prevents the EKF from integrating stale speed into position.

        // ── Step 2: Determine validated speed ───────────────────────────
        
        if (_confirmedStationary) {
          // Vehicle is confirmed stopped — force speed to zero and correct EKF.
          // Apply ZUPT BEFORE predict so that predict integrates v=0.
          // This is the single most valuable constraint in the DR path: it is
          // what stops the marker creeping while parked at a light, and it
          // needs no model — only accelerometer variance + gyro magnitude.
          _ekf!.updateZupt();
          _mlSpeedMs = 0;
          validatedSpeed = 0.0;
          speedVariance = 0.01; // tight: we are very sure it's stopped
        } else {
          // Vehicle is moving — propagate the speed we had when GNSS dropped.
          //
          // Over a bounded (~30s) outage this beats inferring absolute speed
          // from IMU vibration, because the initial speed is KNOWN exactly
          // from GPS at outage onset. Accelerometer integration was also
          // tried and measured worse (bias leakage compounds over 30s:
          // 43.8% vs 33.3% mean drift on test), so the forward accelerometer
          // is deliberately NOT integrated here.
          //
          // Unlike the old ML path this has no dependency on _alignDone, so
          // dead reckoning now works immediately on GNSS loss instead of
          // freezing the marker whenever phone→vehicle alignment had not yet
          // converged.
          validatedSpeed = _drHoldSpeedMs;
          _mlSpeedMs = validatedSpeed;
          // Moderate confidence: right at outage onset this is essentially a
          // GPS measurement, but it goes stale as the vehicle accelerates or
          // brakes, so don't let the EKF lock onto it as hard as a ZUPT.
          speedVariance = 1.0;
        }

        // ── Step 3: Feed validated speed into EKF as measurement ─────────
        // This corrects the EKF internal speed state before predict().
        _ekf!.updateSpeed(validatedSpeed, variance: speedVariance);

        // ── Step 3b: OSM road aiding ──────────────────────────────────────
        // The road network gives an ABSOLUTE heading reference (~1 deg in
        // replay testing) to replace gyro/compass heading (~25 deg error),
        // and snapping removes cross-track error. Only while moving — a
        // stationary vehicle must not be dragged along a road.
        _roadAidActive = false;
        if (!_confirmedStationary) {
          final s0 = _ekf!.state;
          final m = _roads.match(s0.lat, s0.lon, s0.heading);
          if (m != null) {
            _roadAidActive = true;
            _lastRoadSnapDistM = m.distanceM;
            _ekf!.updateHeading(m.bearingRad, variance: 0.02);
            // Lateral snap. The projection is perpendicular to the road, so
            // this corrects cross-track error without inventing along-track
            // progress — distance travelled still comes from DR.
            _ekf!.updateGnss(m.lat, m.lon);
          }
        }

        // ── Step 4: Complementary heading correction ──────────────────────
        // Gently pull gyro-derived yaw rate towards compass absolute heading.
        //
        // SKIPPED when a road match is active: the road bearing is far more
        // accurate than an in-car compass, and applying both makes things
        // WORSE, not better — measured in replay, the compass term dragged
        // the heading straight back off the road (187 m vs 173 m error).
        // Road wins when available; compass is the fallback.
        double correctedYawRate = _vehYawRate;
        if (!_roadAidActive) {
          double hErr = _compassRad - _ekf!.state.heading;
          while (hErr >  math.pi) hErr -= 2 * math.pi;
          while (hErr < -math.pi) hErr += 2 * math.pi;
          // Proportional gain 0.5 rad/s per rad error (soft pull, not hard lock)
          correctedYawRate = _vehYawRate + hErr * 0.5;
        }

        // ── Step 5: Predict — EKF state speed is now validated ────────────
        // predict() uses the speed already in _x[3] (corrected by updateSpeed
        // above). If speed is 0, position does not change.
        _ekf!.predict(dt, correctedYawRate);
        _ekf!.updateNhc();
      }

      // ── Step 6: Read EKF state ───────────────────────────────────────────
      final s = _ekf!.state;
      currentState = s;

      double targetLat = s.lat;
      double targetLon = s.lon;

      // ── Step 7: Map Matching — ONLY when actually moving ─────────────────
      // CRITICAL: Map matching must NEVER create motion.
      // We only snap position if:
      //   a) GNSS is off (DR mode)
      //   b) A route exists
      //   c) The vehicle is confirmed moving (speed > threshold)
      // This prevents snapping to a road when stationary and making the
      // marker appear to travel along the road.
      if (false && !gnssOn && routePoints.isNotEmpty && !_confirmedStationary && _mlSpeedMs > 0.5) {
        final snapped = MapMatcher.snapToRoute(
          LatLng(targetLat, targetLon),
          routePoints,
          maxDistanceMeters: 50.0,
        );
        if (snapped != null) {
          targetLat = snapped.latitude;
          targetLon = snapped.longitude;
        }
      }

      // ── Step 8: Safety gate — reject unreasonably large position jumps ───
      // If the EKF produces a position jump > 200m in one tick (impossible at
      // any realistic vehicle speed with a 100ms timestep), hold position.
      // This catches NaN propagation, matrix inversion failures, etc.
      const double maxJumpDeg = 0.002; // ~200m in lat/lon degrees
      
      final oldPdrLat = pdrLat;
      final oldPdrLon = pdrLon;
      String updateSource = "EMA_SMOOTHING";

      if ((targetLat - pdrLat).abs() > maxJumpDeg ||
          (targetLon - pdrLon).abs() > maxJumpDeg) {
        // Large correction: could be GNSS reacquisition or init — snap immediately
        pdrLat = targetLat;
        pdrLon = targetLon;
        updateSource = "SNAP_JUMP";
      } else {
        // Normal update: smooth with EMA to prevent UI jitter
        pdrLat = pdrLat * 0.8 + targetLat * 0.2;
        pdrLon = pdrLon * 0.8 + targetLon * 0.2;
      }
      
      // DIAGNOSTIC LOGGING - MOVEMENT TRACE
      if (!gnssOn) {
        print("[DRIVE_TRACE] ------------------------------------------------");
        print("[DRIVE_TRACE] timestamp: ${DateTime.now().toIso8601String()}");
        print("[DRIVE_TRACE] raw accelerometer: [$_ax, $_ay, $_az]");
        print("[DRIVE_TRACE] raw gyroscope: [$_gx, $_gy, $_gz]");
        print("[DRIVE_TRACE] DR held velocity: $_mlSpeedMs m/s (hold-at-GNSS-loss: $_drHoldSpeedMs)");
        print("[DRIVE_TRACE] speed variance: $speedVariance");
        print("[DRIVE_TRACE] stationary detector: $_confirmedStationary (raw: $_stationaryTicks/$_stationaryThresh)");
        print("[DRIVE_TRACE] road aid: ${_roadAidActive ? 'MATCHED' : 'none'} (snap ${_lastRoadSnapDistM.toStringAsFixed(1)}m, ${_roads.segmentCount} segs cached)");
        print("[DRIVE_TRACE] ZUPT state: ${_confirmedStationary ? 'APPLIED' : 'NOT_APPLIED'}");
        print("[DRIVE_TRACE] EKF velocity: ${_ekf!.state.speed} m/s");
        print("[DRIVE_TRACE] DR velocity: $validatedSpeed m/s");
        print("[DRIVE_TRACE] DR position: [${_ekf!.state.lat}, ${_ekf!.state.lon}]");
        print("[DRIVE_TRACE] map-matched position: [$targetLat, $targetLon]");
        print("[DRIVE_TRACE] final marker position: [$pdrLat, $pdrLon]");
        print("[DRIVE_TRACE] marker delta: [${pdrLat - oldPdrLat}, ${pdrLon - oldPdrLon}]");
        print("[DRIVE_TRACE] source/component: $updateSource | alignDone: $_alignDone");
        print("[DRIVE_TRACE] ------------------------------------------------");
      }

    } else {
      // ── WALKING MODE — unchanged ─────────────────────────────────────────
      _ekf!.predict(dt, _gz);
      _ekf!.updateSpeed(0, variance: 0.5);

      currentState = NavState(
        x: _ekf!.state.x, y: _ekf!.state.y,
        heading: _compassRad,
        speed: _pdrSpeedMs,
        lat: pdrLat, lon: pdrLon,
      );
    }

    // Track
    if (trackPoints.isEmpty ||
        _dist(trackPoints.last.lat, trackPoints.last.lon, pdrLat, pdrLon) > 2) {
      if (trackPoints.isNotEmpty) {
        _totalDistM += _dist(trackPoints.last.lat, trackPoints.last.lon, pdrLat, pdrLon);
      }
      trackPoints.add(TrackPoint(pdrLat, pdrLon));
      if (trackPoints.length > 5000) trackPoints.removeAt(0);
    }

    // ── DATA RECORDER: sample every fusion tick (~10 Hz) ─────────────────
    if (_isRecording) {
      _recordedRows.add(
        '${DateTime.now().toIso8601String()},'
        '$_ax,$_ay,$_az,'
        '$_gx,$_gy,$_gz,'
        '$_gpsLat,$_gpsLon,$_gpsSpeed,$_gpsHeading'
      );
    }

    notifyListeners();
  }

  double _dist(double la1, double lo1, double la2, double lo2) {
    const R = 6371000.0;
    final p1 = la1*math.pi/180, p2 = la2*math.pi/180;
    final dp = (la2-la1)*math.pi/180, dl = (lo2-lo1)*math.pi/180;
    final a = math.sin(dp/2)*math.sin(dp/2) +
              math.cos(p1)*math.cos(p2)*math.sin(dl/2)*math.sin(dl/2);
    return 2*R*math.asin(math.sqrt(a.clamp(0, 1)));
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SEARCH & ROUTING
  // ══════════════════════════════════════════════════════════════════════════
  Future<List<SearchResult>> searchPlace(String q) async {
    if (q.length < 3) return [];
    try {
      final url = Uri.parse(
        'https://nominatim.openstreetmap.org/search'
        '?q=${Uri.encodeComponent(q)}'
        '&format=json&limit=6&countrycodes=in',
      );
      final res = await http.get(url, headers: {'User-Agent': 'WayFinder-IDR/1.0'});
      if (res.statusCode != 200) return [];
      final list = jsonDecode(res.body) as List;
      return list.map((j) => SearchResult(
        displayName: j['display_name'] as String,
        lat: double.parse(j['lat']),
        lon: double.parse(j['lon']),
      )).toList();
    } catch (_) { return []; }
  }

  Future<void> routeTo(SearchResult dest) async {
    destLatLng   = LatLng(dest.lat, dest.lon);
    destName     = dest.displayName.split(',').take(2).join(', ');
    routeLoading = true;
    routePoints.clear();
    notifyListeners();

    try {
      final mode = navMode == NavMode.walking ? 'foot' : 'driving';
      final url  = Uri.parse(
        'https://router.project-osrm.org/route/v1/$mode/'
        '${pdrLon.toStringAsFixed(6)},${pdrLat.toStringAsFixed(6)};'
        '${dest.lon.toStringAsFixed(6)},${dest.lat.toStringAsFixed(6)}'
        '?overview=full&geometries=geojson',
      );
      final res = await http.get(url);
      if (res.statusCode == 200) {
        final data  = jsonDecode(res.body);
        final route = (data['routes'] as List).first;
        final distM = (route['distance'] as num).toDouble();
        final durS  = (route['duration'] as num).toDouble();
        routeDist   = distM > 1000
            ? '${(distM/1000).toStringAsFixed(1)} km'
            : '${distM.toInt()} m';
        routeEta    = '${(durS/60).ceil()} min';
        routePoints = (route['geometry']['coordinates'] as List)
            .map((c) => LatLng((c[1] as num).toDouble(), (c[0] as num).toDouble()))
            .toList();
      }
    } catch (_) {}

    routeLoading = false;
    notifyListeners();
  }

  void clearRoute() {
    routePoints.clear();
    destLatLng = null; destName = ''; routeDist = ''; routeEta = '';
    notifyListeners();
  }
}
