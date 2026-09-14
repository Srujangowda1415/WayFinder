import 'dart:math' as math;
void main() {
    List<double> _motionAccelMags = List.filled(20, 9.81);
    
    final minA = _motionAccelMags.reduce(math.min);
    final maxA = _motionAccelMags.reduce(math.max);
    final meanA = _motionAccelMags.reduce((a, b) => a + b) / _motionAccelMags.length;
    
    // Simulating perfect stationary
    double _gx = 0.0, _gy = 0.0, _gz = 0.0;
    final gyroMag = math.sqrt(_gx*_gx + _gy*_gy + _gz*_gz);

    final accelVarianceSmall = (maxA - minA) < 0.5;
    final nearGravity = meanA > 8.5 && meanA < 10.5;
    final gyroSmall = gyroMag < 0.1;

    print("Stationary Test:");
    print("maxA: \$maxA, minA: \$minA, diff: \${maxA - minA}");
    print("meanA: \$meanA");
    print("gyroMag: \$gyroMag");
    print("accelVarianceSmall: \$accelVarianceSmall");
    print("nearGravity: \$nearGravity");
    print("gyroSmall: \$gyroSmall");
}
