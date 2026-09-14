/// speed_estimator.dart
/// Pure-Dart CNN-GRU inference engine — no TFLite, no native code.
///
/// Mirrors the PyTorch SpeedEstimator architecture exactly:
///   Conv1D(6→16, k=3, same) → ReLU → MaxPool(2)
///   Conv1D(16→32, k=3, same) → ReLU → MaxPool(2)
///   GRU(32→32, 1 layer)
///   Linear(32→1)
///
/// Weights are loaded from assets/model_weights.json (174 KB).
/// Preprocessing is identical to training: alignment then z-score normalisation.

library;

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

class ModelOutput {
  final double speed;
  final double variance;
  final double vibrationScore;

  const ModelOutput({
    required this.speed,
    required this.variance,
    required this.vibrationScore,
  });
}

class SpeedEstimatorModel {
  // Weights
  late List<List<List<double>>> _w1;  // (16, 6, 3)
  late List<double>             _b1;  // (16,)
  late List<List<List<double>>> _w2;  // (32, 16, 3)
  late List<double>             _b2;  // (32,)
  late List<List<double>>       _Wih; // (96, 32)
  late List<List<double>>       _Whh; // (96, 32)
  late List<double>             _bih; // (96,)
  late List<double>             _bhh; // (96,)
  late List<double>             _wfc_spd; // (32,)
  late double                   _bfc_spd;
  late List<double>             _wfc_var; // (32,)
  late double                   _bfc_var;
  late List<double>             _wfc_vib; // (32,)
  late double                   _bfc_vib;

  // Norm stats
  late List<double> _mean; // (6,) aligned: ax, ay, az, gx, gy, gz
  late List<double> _std;  // (6,)

  bool _loaded = false;
  bool get isLoaded => _loaded;

  // Sliding window buffer (50 samples × 6 features), pre-normalised
  final List<List<double>> _window = [];
  static const int windowSize = 50;

  // ── Load ──────────────────────────────────────────────────────────────────
  Future<void> load() async {
    if (_loaded) return;

    // Load weights
    final wStr = await rootBundle.loadString('assets/model_weights.json');
    final wMap = jsonDecode(wStr) as Map<String, dynamic>;

    _w1 = _load3d(wMap['conv1.weight'] as List);  // (16, 6, 3)
    _b1 = _load1d(wMap['conv1.bias'] as List);
    _w2 = _load3d(wMap['conv2.weight'] as List);  // (32, 16, 3)
    _b2 = _load1d(wMap['conv2.bias'] as List);
    _Wih = _load2d(wMap['gru.weight_ih_l0'] as List);  // (96, 32)
    _Whh = _load2d(wMap['gru.weight_hh_l0'] as List);  // (96, 32)
    _bih = _load1d(wMap['gru.bias_ih_l0'] as List);
    _bhh = _load1d(wMap['gru.bias_hh_l0'] as List);

    final wfc2d = wMap['fc.weight'] as List;  // (3, 32)
    _wfc_spd = _load1d(wfc2d[0] as List);
    _wfc_var = _load1d(wfc2d[1] as List);
    _wfc_vib = _load1d(wfc2d[2] as List);
    
    final bfc = wMap['fc.bias'] as List;
    _bfc_spd = bfc[0] as double;
    _bfc_var = bfc[1] as double;
    _bfc_vib = bfc[2] as double;

    // Load norm stats
    final nStr = await rootBundle.loadString('assets/norm_stats.json');
    final nMap = jsonDecode(nStr) as Map<String, dynamic>;

    final keys = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'];
    _mean = keys.map((k) => (nMap[k]['mean'] as num).toDouble()).toList();
    _std  = keys.map((k) => (nMap[k]['std'] as num).toDouble()).toList();

    _loaded = true;
  }

  // ── Push a new IMU sample (raw, vehicle-aligned) ──────────────────────────
  /// Call at 10 Hz from the fusion loop.
  /// [ax, ay, az]: accelerometer m/s² in VEHICLE frame (X=forward, Z=down)
  /// [gx, gy, gz]: gyroscope rad/s in VEHICLE frame
  void pushSample(double ax, double ay, double az,
                  double gx, double gy, double gz) {
    final raw = [ax, ay, az, gx, gy, gz];
    final norm = List<double>.generate(6, (i) => (raw[i] - _mean[i]) / _std[i]);
    _window.add(norm);
    if (_window.length > windowSize) _window.removeAt(0);
  }

  /// Returns predicted forward speed, variance, and vibration score.
  ModelOutput? predictSpeed() {
    if (!_loaded || _window.length < windowSize) return null;
    return _forward(List.from(_window));
  }

  // ── Forward pass ─────────────────────────────────────────────────────────
  ModelOutput _forward(List<List<double>> x) {
    // x: (50, 6)
    var y = _conv1d(x, _w1, _b1, relu: true);   // (50, 16)
    y = _maxpool(y, 2);                           // (25, 16)
    y = _conv1d(y, _w2, _b2, relu: true);        // (25, 32)
    y = _maxpool(y, 2);                           // (12, 32)
    final h = _gru(y);                            // (32,)
    
    double speed = _bfc_spd;
    double logVar = _bfc_var;
    double vib = _bfc_vib;
    
    for (int i = 0; i < 32; i++) {
      speed += h[i] * _wfc_spd[i];
      logVar += h[i] * _wfc_var[i];
      vib += h[i] * _wfc_vib[i];
    }
    
    return ModelOutput(
      speed: math.max(0.0, speed),
      variance: math.exp(logVar), // Convert log-variance to variance
      vibrationScore: vib
    );
  }

  // Conv1d with 'same' padding, optional ReLU
  List<List<double>> _conv1d(List<List<double>> x,
      List<List<List<double>>> w, List<double> b, {required bool relu}) {
    final T = x.length;
    final cIn = x[0].length;
    final cOut = w.length;
    final k = w[0][0].length;
    final pad = k ~/ 2;
    final out = List.generate(T, (_) => List.filled(cOut, 0.0));

    for (int t = 0; t < T; t++) {
      for (int co = 0; co < cOut; co++) {
        double sum = b[co];
        for (int ki = 0; ki < k; ki++) {
          final ti = t - pad + ki;
          if (ti < 0 || ti >= T) continue;
          for (int ci = 0; ci < cIn; ci++) {
            sum += w[co][ci][ki] * x[ti][ci];
          }
        }
        out[t][co] = relu ? (sum > 0 ? sum : 0.0) : sum;
      }
    }
    return out;
  }

  // MaxPool1d
  List<List<double>> _maxpool(List<List<double>> x, int k) {
    final T = x.length;
    final C = x[0].length;
    final T2 = T ~/ k;
    return List.generate(T2, (i) {
      return List.generate(C, (c) {
        double mx = x[i * k][c];
        for (int j = 1; j < k; j++) {
          if (x[i * k + j][c] > mx) mx = x[i * k + j][c];
        }
        return mx;
      });
    });
  }

  // GRU (1 layer, returns final hidden state)
  List<double> _gru(List<List<double>> x) {
    final H = 32;
    final T = x.length;
    var h = List.filled(H, 0.0);

    for (int t = 0; t < T; t++) {
      final xt = x[t];
      // Gates: i_r, i_z, i_n = Wih @ xt + bih (split into 3 × H)
      // Gates: h_r, h_z, h_n = Whh @ h  + bhh
      final gi = List.filled(3 * H, 0.0);
      final gh = List.filled(3 * H, 0.0);
      for (int r = 0; r < 3 * H; r++) {
        double si = _bih[r], sh = _bhh[r];
        for (int c = 0; c < H; c++) {
          si += _Wih[r][c] * xt[c];
          sh += _Whh[r][c] * h[c];
        }
        gi[r] = si;
        gh[r] = sh;
      }
      final hn = List.filled(H, 0.0);
      for (int i = 0; i < H; i++) {
        final r = _sigmoid(gi[i]      + gh[i]);
        final z = _sigmoid(gi[H + i]  + gh[H + i]);
        final n = _tanh(gi[2*H+i] + r * gh[2*H+i]);
        hn[i]   = (1 - z) * n + z * h[i];
      }
      h = hn;
    }
    return h;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x.clamp(-30, 30)));
  static double _tanh(double x) {
    final e = math.exp(x.clamp(-30, 30) * 2);
    return (e - 1) / (e + 1);
  }

  static List<double> _load1d(List l) =>
      l.map((e) => (e as num).toDouble()).toList();

  static List<List<double>> _load2d(List l) =>
      (l as List).map((row) => _load1d(row as List)).toList();

  static List<List<List<double>>> _load3d(List l) =>
      (l as List).map((m) => _load2d(m as List)).toList();
}
