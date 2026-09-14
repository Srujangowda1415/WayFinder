/// compass_widget.dart — Animated compass rose

import 'dart:math' as math;
import 'package:flutter/material.dart';

class CompassWidget extends StatelessWidget {
  final double headingDeg;

  const CompassWidget({super.key, required this.headingDeg});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 70,
      height: 70,
      child: CustomPaint(
        painter: _CompassPainter(headingDeg: headingDeg),
      ),
    );
  }
}

class _CompassPainter extends CustomPainter {
  final double headingDeg;

  _CompassPainter({required this.headingDeg});

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = size.width / 2 - 4;

    // Outer ring
    final ringPaint = Paint()
      ..color = const Color(0xFF1E2D4A)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawCircle(Offset(cx, cy), r, ringPaint);

    // Cardinal directions
    final textPainter = TextPainter(textDirection: TextDirection.ltr);
    const cardinals = ['N', 'E', 'S', 'W'];
    const angles = [0.0, 90.0, 180.0, 270.0];
    for (int i = 0; i < 4; i++) {
      final a = (angles[i] - headingDeg) * math.pi / 180.0;
      final x = cx + (r - 12) * math.sin(a);
      final y = cy - (r - 12) * math.cos(a);
      textPainter.text = TextSpan(
        text: cardinals[i],
        style: TextStyle(
          color: cardinals[i] == 'N'
              ? const Color(0xFFFF4757)
              : const Color(0xFF7B8BA5),
          fontSize: 9,
          fontWeight: FontWeight.w700,
        ),
      );
      textPainter.layout();
      textPainter.paint(
          canvas, Offset(x - textPainter.width / 2, y - textPainter.height / 2));
    }

    // Needle
    final northAngle = -headingDeg * math.pi / 180.0;
    final needleRed = Paint()..color = const Color(0xFFFF4757);
    final needleWhite = Paint()..color = const Color(0xFF7B8BA5);

    final tipX = cx + (r - 18) * math.sin(northAngle);
    final tipY = cy - (r - 18) * math.cos(northAngle);
    final baseX = cx - (r - 22) * math.sin(northAngle);
    final baseY = cy + (r - 22) * math.cos(northAngle);

    final leftX = cx + 5 * math.cos(northAngle);
    final leftY = cy + 5 * math.sin(northAngle);
    final rightX = cx - 5 * math.cos(northAngle);
    final rightY = cy - 5 * math.sin(northAngle);

    canvas.drawPath(
      Path()
        ..moveTo(tipX, tipY)
        ..lineTo(leftX, leftY)
        ..lineTo(rightX, rightY)
        ..close(),
      needleRed,
    );
    canvas.drawPath(
      Path()
        ..moveTo(baseX, baseY)
        ..lineTo(leftX, leftY)
        ..lineTo(rightX, rightY)
        ..close(),
      needleWhite,
    );

    // Center dot
    canvas.drawCircle(Offset(cx, cy), 3,
        Paint()..color = const Color(0xFF1E2D4A));
  }

  @override
  bool shouldRepaint(covariant _CompassPainter old) =>
      old.headingDeg != headingDeg;
}
