"""Silent screen capture via the XDG ScreenCast portal + PipeWire + GStreamer.

Usage:
    cap = ScreenCapture()   # shows one permission dialog, then silent forever
    img = cap.grab()        # returns PIL Image of current screen
    cap.stop()
"""
import random
import string
import time

import dbus
import dbus.mainloop.glib
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp
from PIL import Image


def _token():
    return "sbi" + "".join(random.choices(string.ascii_lowercase, k=6))


class ScreenCapture:
    def __init__(self):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        Gst.init(None)
        self._bus = dbus.SessionBus()
        self._portal = self._bus.get_object(
            "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop"
        )
        self._screencast = dbus.Interface(self._portal, "org.freedesktop.portal.ScreenCast")
        self._node_id = self._negotiate()
        self._pipeline, self._sink = self._build_pipeline(self._node_id)

    # ── Portal negotiation ────────────────────────────────────────────────────

    def _wait_response(self, request_path, timeout=30):
        """Block until the portal responds to request_path, return (code, results)."""
        result = {}
        loop = GLib.MainLoop()

        def on_response(response, results):
            result["code"] = int(response)
            result["data"] = dict(results)
            loop.quit()

        self._bus.add_signal_receiver(
            on_response, "Response",
            dbus_interface="org.freedesktop.portal.Request",
            path=request_path,
        )
        GLib.timeout_add_seconds(timeout, loop.quit)
        loop.run()
        if "code" not in result:
            raise TimeoutError("Portal request timed out")
        if result["code"] != 0:
            raise PermissionError(f"Portal request denied (code {result['code']})")
        return result["data"]

    def _negotiate(self):
        print("Setting up screen capture — a permission dialog will appear once.", flush=True)

        # 1. Create session
        st = _token()
        req = self._screencast.CreateSession(
            dbus.Dictionary({"session_handle_token": st, "handle_token": _token()}, signature="sv")
        )
        r = self._wait_response(req)
        session = r["session_handle"]

        # 2. Select sources (shows UI for user to pick screen/window)
        req = self._screencast.SelectSources(
            session,
            dbus.Dictionary({
                "handle_token": _token(),
                "types": dbus.UInt32(1 | 2),   # 1=monitor 2=window
                "multiple": dbus.Boolean(False),
                "cursor_mode": dbus.UInt32(2),  # embedded cursor
            }, signature="sv"),
        )
        self._wait_response(req)

        # 3. Start — returns PipeWire node ID
        req = self._screencast.Start(
            session, "",
            dbus.Dictionary({"handle_token": _token()}, signature="sv"),
        )
        r = self._wait_response(req)
        node_id = int(r["streams"][0][0])
        print(f"Screen capture ready (PipeWire node {node_id}).", flush=True)
        return node_id

    # ── GStreamer pipeline ────────────────────────────────────────────────────

    def _build_pipeline(self, node_id):
        pipeline = Gst.parse_launch(
            f"pipewiresrc path={node_id} ! "
            "videoconvert ! "
            "video/x-raw,format=RGB ! "
            "appsink name=sink drop=true max-buffers=1 sync=false"
        )
        sink = pipeline.get_by_name("sink")
        pipeline.set_state(Gst.State.PLAYING)
        time.sleep(0.5)  # let pipeline stabilise
        return pipeline, sink

    # ── Public API ────────────────────────────────────────────────────────────

    def grab(self) -> Image.Image | None:
        sample = self._sink.emit("pull-sample")
        if not sample:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        img = Image.frombytes("RGB", (w, h), bytes(info.data))
        buf.unmap(info)
        return img

    def stop(self):
        self._pipeline.set_state(Gst.State.NULL)
