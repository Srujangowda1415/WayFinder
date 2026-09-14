/// home_screen.dart — Google Maps-style navigation UI
/// Clean, minimal, full-screen map with floating elements.

library;

import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';

import '../core/navigation_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final MapController  _map    = MapController();
  final TextEditingController _searchCtrl = TextEditingController();
  final FocusNode _searchFocus = FocusNode();

  late AnimationController _dotPulse;
  late AnimationController _sheetCtrl;

  Timer? _searchDebounce;
  List<SearchResult> _results = [];
  bool   _showResults   = false;
  bool   _mapFollowUser = true;
  bool   _sheetExpanded = false;

  @override
  void initState() {
    super.initState();
    _dotPulse = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();
    _sheetCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 300), value: 0);
    _searchFocus.addListener(() {
      if (!_searchFocus.hasFocus) setState(() => _showResults = false);
    });
  }

  @override
  void dispose() {
    _dotPulse.dispose(); _sheetCtrl.dispose();
    _searchCtrl.dispose(); _searchFocus.dispose();
    _searchDebounce?.cancel();
    super.dispose();
  }

  // ── Search ─────────────────────────────────────────────────────────────────
  void _onSearchType(String v) {
    _searchDebounce?.cancel();
    if (v.length < 3) { setState(() { _results = []; _showResults = false; }); return; }
    _searchDebounce = Timer(const Duration(milliseconds: 500), () async {
      final nav = context.read<NavigationService>();
      final r   = await nav.searchPlace(v);
      if (mounted) setState(() { _results = r; _showResults = r.isNotEmpty; });
    });
  }

  void _pickResult(SearchResult r) {
    _searchCtrl.text = r.displayName.split(',').take(2).join(', ');
    _searchFocus.unfocus();
    setState(() { _showResults = false; _results = []; });
    final nav = context.read<NavigationService>();
    if (!nav.isRunning) nav.start();
    nav.routeTo(r);
    // Fly map to destination
    _map.move(LatLng(r.lat, r.lon), 14);
    _mapFollowUser = false;
  }

  void _clearSearch() {
    _searchCtrl.clear();
    _results = [];
    _showResults = false;
    context.read<NavigationService>().clearRoute();
  }

  // ── Map moves ──────────────────────────────────────────────────────────────
  void _followUser(NavigationService nav) {
    if (!_mapFollowUser || nav.currentState == null) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _map.move(LatLng(nav.pdrLat, nav.pdrLon), _map.camera.zoom);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<NavigationService>(builder: (_, nav, __) {
      _followUser(nav);
      return Scaffold(
        backgroundColor: const Color(0xFF0A0D14),
        resizeToAvoidBottomInset: false,
        body: GestureDetector(
          onTap: () => _searchFocus.unfocus(),
          child: Stack(children: [
            _buildMap(nav),
            _buildTopBar(nav),
            if (_showResults) _buildDropdown(),
            _buildRightFABs(nav),
            _buildBottomSheet(nav),
          ]),
        ),
      );
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // MAP
  // ══════════════════════════════════════════════════════════════════════════
  Widget _buildMap(NavigationService nav) {
    final center = LatLng(nav.pdrLat, nav.pdrLon);
    final track  = nav.trackPoints.map((p) => LatLng(p.lat, p.lon)).toList();

    return FlutterMap(
      mapController: _map,
      options: MapOptions(
        initialCenter: center,
        initialZoom: 16,
        maxZoom: 19,
        minZoom: 3,
        onTap: (_, __) { setState(() => _mapFollowUser = false); _searchFocus.unfocus(); },
      ),
      children: [
        // Dark base tiles
        TileLayer(
          urlTemplate: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
          subdomains: const ['a', 'b', 'c', 'd'],
          userAgentPackageName: 'com.wayfinder.app',
        ),

        // Route
        if (nav.routePoints.isNotEmpty)
          PolylineLayer<Object>(polylines: [
            Polyline(points: nav.routePoints, strokeWidth: 6, color: const Color(0xFF4285F4)),
          ]),

        // Walked trajectory
        if (track.length > 1)
          PolylineLayer<Object>(polylines: [
            Polyline(points: track, strokeWidth: 3, color: const Color(0xFF34A853).withOpacity(0.7)),
          ]),

        // Accuracy circle
        if (nav.posUncertaintyM < 200)
          CircleLayer(circles: [
            CircleMarker(
              point: center,
              radius: nav.posUncertaintyM,
              useRadiusInMeter: true,
              color: const Color(0x224285F4),
              borderColor: const Color(0x884285F4),
              borderStrokeWidth: 1,
            ),
          ]),

        // Blue dot (position marker)
        MarkerLayer(markers: [
          Marker(
            point: center,
            width: 60, height: 60,
            child: _buildBlueDot(nav),
          ),
        ]),

        // Destination pin
        if (nav.destLatLng != null)
          MarkerLayer(markers: [
            Marker(
              point: nav.destLatLng!,
              width: 40, height: 48,
              child: const Icon(Icons.location_pin, color: Color(0xFFEA4335), size: 42),
            ),
          ]),
      ],
    );
  }

  /// Google Maps-style blue dot with heading arrow and pulsing accuracy ring.
  Widget _buildBlueDot(NavigationService nav) {
    return AnimatedBuilder(
      animation: _dotPulse,
      builder: (_, __) {
        final pulse = (math.sin(_dotPulse.value * 2 * math.pi) + 1) / 2;
        return Stack(alignment: Alignment.center, children: [
          // Pulsing ring
          Container(
            width: 44 + pulse * 12,
            height: 44 + pulse * 12,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Color.fromRGBO(66, 133, 244, 0.15 + pulse * 0.15), width: 2),
            ),
          ),
          // Heading cone
          Transform.rotate(
            angle: nav.headingDeg * math.pi / 180,
            child: CustomPaint(size: const Size(44, 44), painter: _HeadingCone()),
          ),
          // Blue dot
          Container(
            width: 18, height: 18,
            decoration: BoxDecoration(
              color: const Color(0xFF4285F4),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 2.5),
              boxShadow: const [BoxShadow(color: Color(0x664285F4), blurRadius: 8, spreadRadius: 2)],
            ),
          ),
        ]);
      },
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TOP BAR (Search)
  // ══════════════════════════════════════════════════════════════════════════
  Widget _buildTopBar(NavigationService nav) {
    final top = MediaQuery.of(context).padding.top;
    return Positioned(
      top: top + 10, left: 12, right: 12,
      child: Column(children: [
        // Search card
        Material(
          elevation: 6,
          borderRadius: BorderRadius.circular(14),
          color: const Color(0xFF1C2030),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
            child: Row(children: [
              // Back / hamburger
              if (nav.routePoints.isNotEmpty)
                GestureDetector(
                  onTap: _clearSearch,
                  child: const Icon(Icons.arrow_back, color: Colors.white70, size: 22),
                )
              else
                const Icon(Icons.search, color: Colors.white38, size: 22),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: _searchCtrl,
                  focusNode: _searchFocus,
                  onChanged: _onSearchType,
                  style: const TextStyle(color: Colors.white, fontSize: 15),
                  decoration: InputDecoration(
                    hintText: nav.routePoints.isNotEmpty
                        ? nav.destName
                        : 'Search in ${nav.initCity.isEmpty ? "Bengaluru" : nav.initCity}',
                    hintStyle: const TextStyle(color: Colors.white38, fontSize: 14),
                    border: InputBorder.none,
                    isDense: true,
                  ),
                ),
              ),
              if (_searchCtrl.text.isNotEmpty)
                GestureDetector(
                  onTap: _clearSearch,
                  child: const Icon(Icons.close, color: Colors.white38, size: 20),
                )
              else if (nav.routeLoading)
                const SizedBox(width: 20, height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF4285F4))),
            ]),
          ),
        ),

        // Route info chip
        if (nav.routePoints.isNotEmpty && !_showResults)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Material(
              elevation: 4,
              borderRadius: BorderRadius.circular(10),
              color: const Color(0xFF4285F4),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                child: Row(children: [
                  const Icon(Icons.directions, color: Colors.white, size: 18),
                  const SizedBox(width: 8),
                  Text('${nav.routeEta}  ·  ${nav.routeDist}',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 14)),
                  const Spacer(),
                  Text(nav.navMode == NavMode.walking ? '🚶' : '🚗',
                    style: const TextStyle(fontSize: 16)),
                ]),
              ),
            ),
          ),

        // Status chip (init)
        if (nav.initStatus == InitStatus.loading)
          _chip(Icons.my_location, 'Locating you…', const Color(0xFF4285F4)),
        if (nav.initStatus == InitStatus.ipLocated && nav.gnssStatus == GnssStatus.denied && !nav.isRunning)
          _chip(Icons.wifi, 'Located via network — tap Start', const Color(0xFF34A853)),
        if (nav.gnssSimDenied && nav.isRunning)
          _chip(Icons.gps_off, 'GNSS denied — Dead Reckoning ON', const Color(0xFFEA4335)),
      ]),
    );
  }

  Widget _chip(IconData icon, String text, Color color) {
    return Container(
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, color: color, size: 13),
        const SizedBox(width: 6),
        Text(text, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
      ]),
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SEARCH DROPDOWN
  // ══════════════════════════════════════════════════════════════════════════
  Widget _buildDropdown() {
    final top = MediaQuery.of(context).padding.top + 62;
    return Positioned(
      top: top, left: 12, right: 12,
      child: Material(
        elevation: 8,
        borderRadius: BorderRadius.circular(14),
        color: const Color(0xFF1C2030),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(14),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 280),
            child: ListView.builder(
              shrinkWrap: true,
              padding: EdgeInsets.zero,
              itemCount: _results.length,
              itemBuilder: (_, i) {
                final r = _results[i];
                final parts  = r.displayName.split(', ');
                final title  = parts.first;
                final sub    = parts.skip(1).take(3).join(', ');
                return Column(children: [
                  if (i > 0) const Divider(height: 1, color: Color(0xFF252A3A)),
                  ListTile(
                    leading: const Icon(Icons.place_outlined, color: Color(0xFF4285F4), size: 20),
                    title: Text(title, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600), maxLines: 1, overflow: TextOverflow.ellipsis),
                    subtitle: sub.isEmpty ? null : Text(sub, style: const TextStyle(color: Colors.white38, fontSize: 11), maxLines: 1, overflow: TextOverflow.ellipsis),
                    dense: true,
                    onTap: () => _pickResult(r),
                  ),
                ]);
              },
            ),
          ),
        ),
      ),
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // RIGHT FABS (compass, zoom, locate)
  // ══════════════════════════════════════════════════════════════════════════
  Widget _buildRightFABs(NavigationService nav) {
    return Positioned(
      right: 12,
      bottom: _sheetExpanded ? 340 : 180,
      child: Column(children: [
        // ── Record FAB (always visible) ──
        _recFab(nav),
        const SizedBox(height: 8),
        _fab(Icons.add, () => _map.move(_map.camera.center, _map.camera.zoom + 1)),
        const SizedBox(height: 8),
        _fab(Icons.remove, () => _map.move(_map.camera.center, _map.camera.zoom - 1)),
        const SizedBox(height: 8),
        _fab(
          _mapFollowUser ? Icons.gps_fixed : Icons.gps_not_fixed,
          () { setState(() => _mapFollowUser = true); _map.move(LatLng(nav.pdrLat, nav.pdrLon), 16); },
          color: _mapFollowUser ? const Color(0xFF4285F4) : Colors.white70,
        ),
      ]),
    );
  }

  Widget _recFab(NavigationService nav) {
    final rec = nav.isRecording;
    return GestureDetector(
      onTap: () async {
        if (rec) {
          await nav.stopRecording();
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              backgroundColor: const Color(0xFF1C2030),
              content: Text(
                nav.lastRecordingPath.isNotEmpty
                    ? 'Saved: ${nav.lastRecordingPath.split('/').last}'
                    : 'Recording saved.',
                style: const TextStyle(color: Colors.white),
              ),
            ));
          }
        } else {
          nav.startRecording();
        }
      },
      child: AnimatedBuilder(
        animation: _dotPulse,
        builder: (_, __) {
          final pulse = rec ? (math.sin(_dotPulse.value * 2 * math.pi) + 1) / 2 : 0.0;
          return Container(
            width: 48, height: 48,
            decoration: BoxDecoration(
              color: rec
                  ? Color.fromRGBO(234, 67, 53, (0.7 + pulse * 0.3))
                  : const Color(0xFF1C2030),
              shape: BoxShape.circle,
              boxShadow: [BoxShadow(
                color: rec
                    ? Colors.red.withOpacity(0.5 + pulse * 0.4)
                    : Colors.black.withOpacity(0.4),
                blurRadius: rec ? 12 : 6,
              )],
            ),
            child: Icon(
              rec ? Icons.stop : Icons.fiber_manual_record,
              color: rec ? Colors.white : const Color(0xFFEA4335),
              size: 22,
            ),
          );
        },
      ),
    );
  }

  Widget _fab(IconData icon, VoidCallback onTap, {Color color = Colors.white70}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 42, height: 42,
        decoration: BoxDecoration(
          color: const Color(0xFF1C2030),
          shape: BoxShape.circle,
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.4), blurRadius: 6)],
        ),
        child: Icon(icon, color: color, size: 20),
      ),
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // BOTTOM SHEET
  // ══════════════════════════════════════════════════════════════════════════
  Widget _buildBottomSheet(NavigationService nav) {
    final bottom = MediaQuery.of(context).padding.bottom;
    return Positioned(
      bottom: 0, left: 0, right: 0,
      child: GestureDetector(
        onVerticalDragUpdate: (d) {
          if (d.delta.dy < -4) setState(() => _sheetExpanded = true);
          if (d.delta.dy >  4) setState(() => _sheetExpanded = false);
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
          decoration: const BoxDecoration(
            color: Color(0xFF1C2030),
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Padding(
            padding: EdgeInsets.fromLTRB(20, 12, 20, bottom + 16),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              // Handle
              Container(width: 36, height: 4, decoration: BoxDecoration(
                color: Colors.white24, borderRadius: BorderRadius.circular(2))),
              const SizedBox(height: 16),

              // Speed row
              Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(nav.speedKmh.toStringAsFixed(0),
                    style: const TextStyle(color: Colors.white, fontSize: 42, fontWeight: FontWeight.w200, height: 1)),
                  const Text('km/h', style: TextStyle(color: Colors.white38, fontSize: 12)),
                ]),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    _statLabel('${nav.headingDeg.toStringAsFixed(0)}° ${_cardinal(nav.headingDeg)}', 'Head'),
                    const SizedBox(height: 4),
                    _statLabel('${nav.stepCount}', 'Steps'),
                  ]),
                ),
                Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                  Text('${nav.totalDistanceKm.toStringAsFixed(2)} km',
                    style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
                  Text(_fmt(nav.elapsed), style: const TextStyle(color: Colors.white38, fontSize: 12)),
                  if (nav.gnssStatus == GnssStatus.active)
                    Text('GPS ±${nav.gnssAccuracyM.toStringAsFixed(0)}m',
                      style: const TextStyle(color: Color(0xFF34A853), fontSize: 11, fontWeight: FontWeight.w600)),
                ]),
              ]),

              if (_sheetExpanded) ...[
                const SizedBox(height: 16),
                // Mode row
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  alignment: WrapAlignment.spaceBetween,
                  children: [
                    _modeChip('🚶 Walk', nav.navMode == NavMode.walking, () => nav.setNavMode(NavMode.walking)),
                    _modeChip('🚗 Drive', nav.navMode == NavMode.vehicle, () => nav.setNavMode(NavMode.vehicle)),
                    _modeChip(
                      nav.gnssSimDenied ? '📡 GNSS off' : '📡 GNSS on',
                      nav.gnssSimDenied,
                      () => nav.toggleGnssDenied(),
                      activeColor: const Color(0xFFEA4335),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
              ],

              const SizedBox(height: 16),
              // Start / Stop
              SizedBox(width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => nav.isRunning ? nav.stop() : nav.start(),
                  icon: Icon(nav.isRunning ? Icons.stop_rounded : Icons.navigation_rounded, size: 20),
                  label: Text(nav.isRunning ? 'Stop Navigation' : 'Start Navigation',
                    style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: nav.isRunning ? const Color(0xFFEA4335) : const Color(0xFF4285F4),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    elevation: 4,
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _statLabel(String value, String label) {
    return Row(children: [
      Text(value, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
      const SizedBox(width: 4),
      Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
    ]);
  }

  Widget _modeChip(String label, bool active, VoidCallback onTap, {Color activeColor = const Color(0xFF4285F4)}) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: active ? activeColor.withOpacity(0.15) : const Color(0xFF252A3A),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: active ? activeColor : Colors.transparent),
        ),
        child: Text(label, style: TextStyle(
          color: active ? activeColor : Colors.white54,
          fontSize: 12, fontWeight: FontWeight.w600)),
      ),
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────
  String _fmt(Duration d) =>
    '${d.inHours.toString().padLeft(2,'0')}:${(d.inMinutes%60).toString().padLeft(2,'0')}:${(d.inSeconds%60).toString().padLeft(2,'0')}';

  String _cardinal(double deg) {
    const d = ['N','NE','E','SE','S','SW','W','NW'];
    return d[((deg+22.5)/45).floor()%8];
  }
}

/// Custom painter for the heading cone behind the blue dot
class _HeadingCone extends CustomPainter {
  @override
  void paint(Canvas canvas, Size s) {
    final cx = s.width / 2, cy = s.height / 2;
    final paint = Paint()
      ..shader = RadialGradient(
        colors: [const Color(0x554285F4), Colors.transparent],
        stops: const [0.0, 1.0],
      ).createShader(Rect.fromCircle(center: Offset(cx, cy), radius: s.width / 2));
    final path = ui.Path()
      ..moveTo(cx, cy)
      ..lineTo(cx - 14, 0)
      ..lineTo(cx + 14, 0)
      ..close();
    canvas.drawPath(path, paint);
  }
  @override
  bool shouldRepaint(_) => false;
}
