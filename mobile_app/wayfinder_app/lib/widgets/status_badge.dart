/// status_badge.dart — GNSS status indicator

import 'package:flutter/material.dart';
import '../core/navigation_service.dart';

class StatusBadge extends StatelessWidget {
  final GnssStatus status;

  const StatusBadge({super.key, required this.status});

  @override
  Widget build(BuildContext context) {
    final (color, label, icon) = switch (status) {
      GnssStatus.active   => (const Color(0xFF00FF9D), 'GNSS ACTIVE', Icons.gps_fixed),
      GnssStatus.degraded => (const Color(0xFFF0C040), 'GNSS DEGRADED', Icons.gps_not_fixed),
      GnssStatus.denied   => (const Color(0xFFFF4757), 'GNSS DENIED', Icons.gps_off),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Color.fromRGBO(color.red, color.green, color.blue, 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Color.fromRGBO(color.red, color.green, color.blue, 0.5),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 12),
          const SizedBox(width: 5),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.6,
            ),
          ),
        ],
      ),
    );
  }
}
