# Running live_transcribe.py on a new machine

## Prerequisites

You need to be on the Cambridge UDN (or VPN) to reach the Turing server.

## Install dependencies

### System packages (apt)

```bash
sudo apt install \
  python3-gi python3-dbus \
  gstreamer1.0-pipewire gstreamer1.0-plugins-base \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
```

### Python packages (pip)

```bash
pip install -r requirements.txt
```

## Find your audio device number

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Find your microphone in the list and note its index number.

## Run

```bash
python3 live_transcribe.py --device N
```

Replace `N` with your microphone's index. On first run, a one-time permission dialog appears for screen capture — approve it. After that it runs silently.

## Notes

- The Turing transcript server must already be running before you start this script.
- Screen capture (slide screenshots) requires GNOME Wayland with the system packages above. If your desktop does not support the XDG ScreenCast portal, the script prints a warning and continues — transcription still works fine, slides just will not be sent.
- The script posts to `http://172.24.17.90:7103` — this is only reachable on the Cambridge UDN.
