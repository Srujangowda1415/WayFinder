/// ekf_navigation.dart
/// Pure-Dart port of the Python EKF fusion engine.
/// Runs at 10 Hz on-device — no Python runtime needed.
///
/// State: [x, y, heading, speed, heading_bias, speed_bias]
/// Measurements: GNSS position, speed, NHC (lateral = 0), ZUPT

import 'dart:math' as math;

class NavState {
  final double x;        // Easting (m)
  final double y;        // Northing (m)
  final double heading;  // rad (N=0, clockwise)
  final double speed;    // m/s
  final double lat;      // degrees (computed from ENU)
  final double lon;      // degrees (computed from ENU)

  const NavState({
    required this.x,
    required this.y,
    required this.heading,
    required this.speed,
    required this.lat,
    required this.lon,
  });
}

class EKFNavigation {
  // Reference origin
  final double lat0;
  final double lon0;
  final double _mPerDegLat;
  final double _mPerDegLon;

  // State vector [x, y, heading, speed, headingBias, speedBias]
  List<double> _x;

  // Covariance matrix (6x6, stored row-major)
  List<List<double>> _P;

  // Process noise
  final List<List<double>> _Q;

  // Measurement noise
  final double _rGnss = 25.0;   // m²
  final double _rSpeed = 0.25;  // (m/s)²
  final double _rNhc = 0.01;    // NHC tight
  final double _rZupt = 0.01;   // ZUPT tight

  bool gnssAvailable = true;

  EKFNavigation({
    required this.lat0,
    required this.lon0,
    required double initHeading,
    double initSpeed = 0.0,
  })  : _mPerDegLat = 6371000.0 * math.pi / 180.0,
        _mPerDegLon = 6371000.0 *
            math.pi /
            180.0 *
            math.cos(lat0 * math.pi / 180.0),
        _x = [0.0, 0.0, initHeading, initSpeed, 0.0, 0.0],
        // heading_bias P0/Q were 1e-6/1e-8 — tight enough that the state was
        // effectively frozen at 0 regardless of measurements. Now that GPS
        // course-over-ground and road-bearing updates correct heading
        // directly (see NavigationService._onGps / _fuse), the bias state
        // needs enough freedom to actually absorb the phone gyro's real
        // bias via those updates' cross-covariance, so it is already learned
        // by the time a GNSS outage starts.
        _P = _diagMatrix(6, [100.0, 100.0, 0.1, 4.0, 1e-4, 0.25]),
        _Q = _diagMatrix(6, [0.01, 0.01, 1e-4, 0.01, 1e-6, 1e-6]);

  /// Propagate state by one timestep.
  void predict(double dt, double yawRate) {
    final psi = _x[2];
    final v = _x[3];
    final bPsi = _x[4];
    final bV = _x[5];

    final psiDot = yawRate - bPsi;

    final psiNew = (psi + psiDot * dt) % (2 * math.pi);
    final vNew = v; // Constant velocity model

    final dx = vNew * math.sin(psiNew) * dt;
    final dy = vNew * math.cos(psiNew) * dt;

    _x = [
      _x[0] + dx,
      _x[1] + dy,
      psiNew,
      vNew,
      bPsi,
      bV,
    ];

    // Jacobian F
    final F = _eye(6);
    F[0][2] = vNew * math.cos(psiNew) * dt;
    F[0][3] = math.sin(psiNew) * dt;
    F[0][4] = -vNew * math.cos(psiNew) * dt;
    F[1][2] = -vNew * math.sin(psiNew) * dt;
    F[1][3] = math.cos(psiNew) * dt;
    F[1][4] = vNew * math.sin(psiNew) * dt;
    F[2][4] = -dt;
    // Speed is constant, no dependency on bV in predict anymore
    F[3][5] = 0.0;

    // P = F*P*F' + Q
    _P = _matAdd(_matMul(_matMul(F, _P), _matTranspose(F)), _Q);
  }

  /// GNSS position update.
  void updateGnss(double lat, double lon) {
    final measX = (lon - lon0) * _mPerDegLon;
    final measY = (lat - lat0) * _mPerDegLat;

    // H selects x and y
    final H = List.generate(2, (_) => List.filled(6, 0.0));
    H[0][0] = 1.0;
    H[1][1] = 1.0;

    final R = _diagMatrix(2, [_rGnss, _rGnss]);
    final innov = [measX - _x[0], measY - _x[1]];

    _kalmanUpdate(H, R, innov);
  }

  /// Speed measurement update.
  void updateSpeed(double speed, {double variance = 0.25}) {
    final H = [List.filled(6, 0.0)];
    H[0][3] = 1.0;

    // Use dynamically predicted variance from AI, or default fallback
    final R = [[variance]];
    final innov = [speed - _x[3]];

    _kalmanUpdate(H, R, innov);
  }

  /// Non-Holonomic Constraint: zero lateral (body-frame right) velocity.
  ///
  /// vRight = vx*cos(psi) - vy*sin(psi), where vx = v*sin(psi), vy = v*cos(psi)
  /// are this state's own forward-velocity components. Substituting gives
  /// vRight ≡ 0 for every psi and v: this state has no independent lateral-
  /// velocity degree of freedom, so the correctly-derived constraint (and its
  /// Jacobian) is an exact no-op here (zero H → zero Kalman gain).
  ///
  /// (The previous version paired the sin/cos terms the wrong way —
  /// `-v*sin(psi)*sin(psi) + v*cos(psi)*cos(psi)` = `v*cos(2*psi)` — which is
  /// generally non-zero and was injecting spurious corrections into the
  /// heading/speed state on every tick.)
  void updateNhc() {
    final psi = _x[2];
    final v = _x[3];

    final H = [List.filled(6, 0.0)];
    H[0][2] = 0.0; // d(vRight)/d(psi) — vRight is identically 0 in this model
    H[0][3] = 0.0; // d(vRight)/d(v)

    final vx = v * math.sin(psi);
    final vy = v * math.cos(psi);
    final vRight = vx * math.cos(psi) - vy * math.sin(psi);
    final innov = [0.0 - vRight];

    final R = [[_rNhc]];
    _kalmanUpdate(H, R, innov);
  }

  /// Zero Velocity Update when vehicle is stationary.
  void updateZupt() {
    final H = [List.filled(6, 0.0)];
    H[0][3] = 1.0;

    final innov = [0.0 - _x[3]];
    final R = [[_rZupt]];

    _kalmanUpdate(H, R, innov);
  }

  /// Compass heading measurement update.
  /// Anchors the EKF heading to the magnetometer-derived azimuth,
  /// preventing gyro integration drift during GNSS outages.
  void updateHeading(double headingRad, {double variance = 0.05}) {
    final H = [List.filled(6, 0.0)];
    H[0][2] = 1.0; // measurement is heading (state index 2)

    // Compute smallest angular error (handles 0/2π wraparound)
    double innov = headingRad - _x[2];
    while (innov >  math.pi) innov -= 2 * math.pi;
    while (innov < -math.pi) innov += 2 * math.pi;

    final R = [[variance]];
    _kalmanUpdate(H, R, [innov]);
  }

  void _kalmanUpdate(
      List<List<double>> H, List<List<double>> R, List<double> innov) {
    final Ht = _matTranspose(H);
    final S = _matAdd(_matMul(_matMul(H, _P), Ht), R);
    final Sinv = _matInv(S);
    final K = _matMul(_matMul(_P, Ht), Sinv);

    // x = x + K*innov
    for (int i = 0; i < 6; i++) {
      double sum = 0.0;
      for (int j = 0; j < innov.length; j++) {
        sum += K[i][j] * innov[j];
      }
      _x[i] += sum;
    }

    // P = (I - K*H)*P
    final KH = _matMul(K, H);
    final IminusKH = _matSub(_eye(6), KH);
    _P = _matMul(IminusKH, _P);
  }

  NavState get state {
    final lat = lat0 + _x[1] / _mPerDegLat;
    final lon = lon0 + _x[0] / _mPerDegLon;
    return NavState(
      x: _x[0],
      y: _x[1],
      heading: _x[2],
      speed: _x[3],
      lat: lat,
      lon: lon,
    );
  }

  double get headingDeg => _x[2] * 180.0 / math.pi;
  double get speedMs => _x[3];
  double get speedKmh => _x[3] * 3.6;
  double get positionUncertaintyM => math.sqrt(_P[0][0] + _P[1][1]);

  // ── Matrix helpers ─────────────────────────────────────────────────────────

  static List<List<double>> _diagMatrix(int n, List<double> diag) {
    final m = List.generate(n, (_) => List.filled(n, 0.0));
    for (int i = 0; i < n; i++) {
      m[i][i] = diag[i];
    }
    return m;
  }

  static List<List<double>> _eye(int n) {
    return _diagMatrix(n, List.filled(n, 1.0));
  }

  static List<List<double>> _matMul(
      List<List<double>> A, List<List<double>> B) {
    final rows = A.length;
    final cols = B[0].length;
    final inner = B.length;
    final C = List.generate(rows, (_) => List.filled(cols, 0.0));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        for (int k = 0; k < inner; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  static List<List<double>> _matTranspose(List<List<double>> A) {
    final rows = A.length;
    final cols = A[0].length;
    return List.generate(cols, (j) => List.generate(rows, (i) => A[i][j]));
  }

  static List<List<double>> _matAdd(
      List<List<double>> A, List<List<double>> B) {
    return List.generate(
        A.length, (i) => List.generate(A[0].length, (j) => A[i][j] + B[i][j]));
  }

  static List<List<double>> _matSub(
      List<List<double>> A, List<List<double>> B) {
    return List.generate(
        A.length, (i) => List.generate(A[0].length, (j) => A[i][j] - B[i][j]));
  }

  /// 2×2 matrix inverse (sufficient for our measurement updates).
  static List<List<double>> _matInv(List<List<double>> A) {
    if (A.length == 1) {
      return [[1.0 / A[0][0]]];
    }
    if (A.length == 2) {
      final det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
      return [
        [A[1][1] / det, -A[0][1] / det],
        [-A[1][0] / det, A[0][0] / det],
      ];
    }
    // For larger matrices, use Gaussian elimination
    final n = A.length;
    final aug = List.generate(n, (i) {
      final row = List<double>.from(A[i]);
      for (int j = 0; j < n; j++) {
        row.add(i == j ? 1.0 : 0.0);
      }
      return row;
    });

    for (int col = 0; col < n; col++) {
      int pivot = col;
      for (int row = col + 1; row < n; row++) {
        if (aug[row][col].abs() > aug[pivot][col].abs()) pivot = row;
      }
      final tmp = aug[col];
      aug[col] = aug[pivot];
      aug[pivot] = tmp;

      final denom = aug[col][col];
      for (int j = 0; j < 2 * n; j++) {
        aug[col][j] /= denom;
      }
      for (int row = 0; row < n; row++) {
        if (row == col) continue;
        final factor = aug[row][col];
        for (int j = 0; j < 2 * n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }

    return List.generate(n, (i) => aug[i].sublist(n));
  }
}
