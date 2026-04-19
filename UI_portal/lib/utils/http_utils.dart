import 'dart:async';
import 'package:http/http.dart' as http;
import 'package:RouteIQ_UI/utils/logger.dart';

// ─── SAFE HTTP HELPERS ────────────────────────────────────────────────────────
//
// Drop-in wrappers that absorb every network failure mode Flutter Web can
// produce (ClientException, TimeoutException, SocketException, etc.) and
// return null instead of crashing.
//
// ┌────────────────┬────────────────────────────────────────────────────┐
// │ safeGet        │ GET  — null for non-2xx AND exceptions             │
// │ safePost       │ POST — null for non-2xx AND exceptions             │
// │ safeFetch      │ GET  — null ONLY on exception; caller owns         │
// │                │        status-code logic (use when you need        │
// │                │        to inspect 425 / 404 / etc.)                │
// └────────────────┴────────────────────────────────────────────────────┘
//
// All helpers apply a configurable timeout and log failures via logDebug.
// ─────────────────────────────────────────────────────────────────────────────

/// GET — returns the response only for 2xx; null otherwise.
///
/// Use for fire-and-forget polling where any non-2xx means "skip this tick"
/// (dashboard, pipeline health, solver status).
Future<http.Response?> safeGet(
  String url, {
  String tag = 'HTTP',
  Duration timeout = const Duration(seconds: 10),
}) async {
  try {
    final res = await http.get(Uri.parse(url)).timeout(timeout);
    if (res.statusCode >= 200 && res.statusCode < 300) return res;
    logDebug('$tag: GET $url → ${res.statusCode}');
    return null;
  } on TimeoutException {
    logDebug('$tag: GET $url → timeout after ${timeout.inSeconds}s');
    return null;
  } catch (e) {
    logDebug('$tag: GET $url → $e');
    return null;
  }
}

/// POST — returns the response only for 2xx; null otherwise.
///
/// Pass [body] as Map<String, String> for form-encoded payloads.
Future<http.Response?> safePost(
  String url, {
  Map<String, String>? body,
  String tag = 'HTTP',
  Duration timeout = const Duration(seconds: 15),
}) async {
  try {
    final res = await http.post(Uri.parse(url), body: body).timeout(timeout);
    if (res.statusCode >= 200 && res.statusCode < 300) return res;
    logDebug('$tag: POST $url → ${res.statusCode}');
    return null;
  } on TimeoutException {
    logDebug('$tag: POST $url → timeout after ${timeout.inSeconds}s');
    return null;
  } catch (e) {
    logDebug('$tag: POST $url → $e');
    return null;
  }
}

/// GET — returns the response for ANY status code; null only on exception.
///
/// Use when you need to inspect non-2xx codes yourself — e.g. HTTP 425
/// ("still computing, retry") or HTTP 404 ("job not found"). The caller is
/// responsible for all status-code branching.
///
/// Example:
///   final res = await safeFetch('$_kBaseUrl/benchmark/$jobId',
///                               tag: 'Benchmark');
///   if (res == null)            { /* network down — bail  */ return; }
///   if (res.statusCode == 200)  { /* parse result         */ }
///   if (res.statusCode == 425)  { /* still running, retry */ }
///   else                        { /* unexpected error      */ }
Future<http.Response?> safeFetch(
  String url, {
  String tag = 'HTTP',
  Duration timeout = const Duration(seconds: 20),
}) async {
  try {
    final res = await http.get(Uri.parse(url)).timeout(timeout);
    return res; // status inspection left to the caller
  } on TimeoutException {
    logDebug('$tag: GET $url → timeout after ${timeout.inSeconds}s');
    return null;
  } catch (e) {
    logDebug('$tag: GET $url → $e');
    return null;
  }
}
