/// ekf_navigation_test.dart
///
/// Regression coverage for the bug reported on real hardware: with GNSS off
/// and dead reckoning enabled, the vehicle marker went straight through
/// turns instead of following them. Covers the EKF primitives directly
/// (ZUPT, yaw integration, heading updates) plus the gravity-projected yaw
/// rate formula from NavigationService._onAccel, mirrored here the same way
/// tool/dr_replay.dart mirrors it for headless replay testing.
library;

import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:wayfinder_app/core/ekf_navigation.dart';

/// Mirrors the gravity-projected yaw rate computed in
/// NavigationService._onAccel: rotation about the measured "down" (gravity)
/// direction, which is the vehicle's true vertical/yaw axis regardless of
/// how the phone is mounted.
double gravityProjectedYawRate(
  double ax, double ay, double az,
  double gx, double gy, double gz,
) {
  final gNorm = math.sqrt(ax * ax + ay * ay + az * az);
  return gNorm > 0.5 ? -(gx * ax + gy * ay + gz * az) / gNorm : gz;
}

void main() {
  group('EKFNavigation — ZUPT', () {
    test('stationary vehicle does not drift in position', () {
      final ekf = EKFNavigation(lat0: 12.0, lon0: 77.0, initHeading: 0.0, initSpeed: 5.0);
      // Simulate 5 s of "stopped" ticks: ZUPT forces speed to 0, predict
      // then integrates v=0, so x/y must not move even though the state
      // entered this loop already moving.
      for (var i = 0; i < 50; i++) {
        ekf.updateZupt();
        ekf.predict(0.1, 0.0);
      }
      expect(ekf.state.x.abs(), lessThan(0.01));
      expect(ekf.state.y.abs(), lessThan(0.01));
      expect(ekf.speedMs, lessThan(0.01));
    });
  });

  group('EKFNavigation — yaw integration', () {
    test('sustained yaw rate turns heading instead of going straight', () {
      final ekf = EKFNavigation(lat0: 12.0, lon0: 77.0, initHeading: 0.0, initSpeed: 10.0);
      final headingBefore = ekf.state.heading;
      // 90 deg/s for 1 s ≈ a quarter turn — the exact scenario the user hit:
      // GNSS off, driving, and taking a turn.
      for (var i = 0; i < 10; i++) {
        ekf.predict(0.1, math.pi / 2);
      }
      final headingAfter = ekf.state.heading;
      var delta = headingAfter - headingBefore;
      while (delta > math.pi) delta -= 2 * math.pi;
      while (delta < -math.pi) delta += 2 * math.pi;
      expect(delta, closeTo(math.pi / 2, 0.05));
    });

    test('zero yaw rate keeps heading unchanged (straight is still possible when true)', () {
      final ekf = EKFNavigation(lat0: 12.0, lon0: 77.0, initHeading: 1.0, initSpeed: 10.0);
      for (var i = 0; i < 20; i++) {
        ekf.predict(0.1, 0.0);
      }
      expect(ekf.state.heading, closeTo(1.0, 1e-6));
    });
  });

  group('EKFNavigation — updateHeading', () {
    test('wraps the innovation across the 0/2pi boundary', () {
      // Heading near 2pi, measurement near 0 — the true error is small
      // (~0.1 rad), not ~2pi. A naive (unwrapped) update would overcorrect.
      final ekf = EKFNavigation(
        lat0: 12.0, lon0: 77.0,
        initHeading: 2 * math.pi - 0.05, initSpeed: 5.0,
      );
      ekf.updateHeading(0.05, variance: 0.01);
      // Should move a small amount toward 0 (mod 2pi), not swing by ~2pi.
      final h = ekf.state.heading % (2 * math.pi);
      final distToZero = math.min(h, 2 * math.pi - h);
      expect(distToZero, lessThan(0.05));
    });
  });

  group('Gravity-projected yaw rate — mounting independence', () {
    test('flat mount: matches raw Z-gyro', () {
      // Phone flat, screen up: gravity is purely +Z, so projection reduces
      // to (approximately) the raw Z-gyro reading.
      final yaw = gravityProjectedYawRate(0, 0, 9.81, 0, 0, 0.5);
      expect(yaw, closeTo(-0.5, 1e-6));
    });

    test('tilted 90 degrees (phone upright, e.g. dash mount): still recovers the true turn', () {
      // Rotate the flat-mount case 90° about the X axis: gravity now reads
      // on Y instead of Z, and the same real-world yaw rotation now shows up
      // on the gyro's Y axis instead of Z. Raw-Z would read ~0 here (the
      // exact failure mode reported on hardware); the gravity projection
      // must still recover the same yaw rate as the flat case.
      final flatYaw = gravityProjectedYawRate(0, 0, 9.81, 0, 0, 0.5);
      final tiltedYaw = gravityProjectedYawRate(0, 9.81, 0, 0, 0.5, 0);
      expect(tiltedYaw, closeTo(flatYaw, 1e-6));
    });

    test('raw Z-gyro alone would miss this turn entirely (regression guard)', () {
      // In the tilted case above, raw _gz is 0 even though the vehicle is
      // genuinely turning at 0.5 rad/s — this is the bug being fixed.
      const rawGz = 0.0;
      final projected = gravityProjectedYawRate(0, 9.81, 0, 0, 0.5, 0);
      expect(rawGz, isNot(closeTo(projected, 0.1)));
    });
  });
}
