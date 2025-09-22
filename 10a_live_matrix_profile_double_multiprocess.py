#!/usr/bin/env python3
"""
Fast, smooth live plotting with fastplotlib + real-time DTW in a separate process.

Key ideas
---------
• DTW runs in a *separate process* (multiprocessing, "spawn" context) so it doesn't fight the UI for the GIL.
• The DTW process consumes only the *latest* snapshot (coalesced queue, maxsize=1).
• Ring buffers (deques) avoid per-frame reallocations.
• UI updates are rate-limited; we *slide* x-limits instead of autoscaling each frame.

Requirements
------------
  pip install fastplotlib dtaidistance numpy requests python-dotenv

Environment
-----------
  PHYPHOX_ADDRESS: host:port of your Phyphox endpoint (e.g., "192.168.1.5:8080")

Usage
-----
  python fastplotlib_realtime_dtw.py \
      --reference-file label_studio_export/project-2-at-2025-07-26-03-40-4a2f6183.json \
      --plot-window 10 --dtw-buffer 1000 --dtw-threshold 9 --min-match-time 0.75 --dtw-channel 0

Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Queue, Full, Empty
from threading import Thread
from types import SimpleNamespace

import numpy as np
import requests
from dtaidistance.subsequence.dtw import SubsequenceAlignment

from dotenv import load_dotenv

# ----------------------------
# Helpers & configuration
# ----------------------------

def _default_phone_url() -> str:
    # Assumes load_dotenv() has already run in main() before Settings() is created.
    addr = os.getenv("PHYPHOX_ADDRESS")
    if not addr:
        raise EnvironmentError("PHYPHOX_ADDRESS environment variable is not set. Please set it in your environment or .env file.")
    return addr if addr.startswith(("http://", "https://")) else f"http://{addr}"

@dataclass
class Settings:
    phone_url: str = field(default_factory=_default_phone_url)
    time_channel: str = "acc_time"
    data_channels: tuple[str, ...] = ("acc", "acc_time")
    plot_window: float = 10.0
    max_buffer_size: int = 1000
    dtw_buffer_size: int = 1000
    dtw_threshold: float = 0.06
    approximate_timestep: float = 0.01
    min_match_time: float = 0.90
    dtw_channel_index: int = 0  # which series goes to DTW (0 => first channel)
    dtw_detection_blackout_sec: float = 1.5 # how long must elapse after the end of the detected step before we confirm it (so short reference steps aren't favoerd over long)
    ref_sequence_filter_and_scale = { # higher scale factor makes more likely to detect
        'Cleese':2.5,
        'Palin':0.8,
        'Forward Walk':1.0,
    }

    @property
    def min_match_len(self) -> int:
        return max(1, int(round(self.min_match_time / max(self.approximate_timestep, 1e-6))))


# coalescing put: keep only the latest item
def put_latest(q: Queue, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except Full:
            try:
                q.get_nowait()  # drop stale
            except Empty:
                pass


# ----------------------------
# Reference sequence loading
# ----------------------------

def _nearest_idx(times: np.ndarray, t: float) -> int:
    # robust nearest by absolute difference
    return int(np.argmin(np.abs(times - t)))


def load_ref_sequences(reference_file: str, settings: Settings) -> tuple[list[np.ndarray], list[str]]:
    """Load timeseries windows + labels from a Label Studio export JSON.

    Expected shape (simplified):
        data[0]['data']['ts']['acc']      -> list/array of values
        data[0]['data']['ts']['acc_time'] -> list/array of times (same length)
        data[0]['annotations'][0]['result'] -> regions with 'value': {'start', 'end', 'timeserieslabels': [label]}
    """
    with open(reference_file, "r") as f:
        ref_data = json.load(f)

    ref_sequence_filter_list = [k for k in settings.ref_sequence_filter_and_scale.keys()]

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    scale_factors: list[float] = []
    for i in range(len(ref_data)):

        acc = np.asarray(ref_data[i]["data"]["ts"]["acc"], dtype=float)
        acc_time = np.asarray(ref_data[i]["data"]["ts"]["acc_time"], dtype=float)

        annotations = ref_data[i]["annotations"][0]["result"]
        for a in annotations:
            v = a["value"]
            t0 = float(v["start"])  # seconds
            t1 = float(v["end"])    # seconds
            label = str(v["timeserieslabels"][0])
            i0 = _nearest_idx(acc_time, t0)
            i1 = _nearest_idx(acc_time, t1)
            if i1 < i0:
                i0, i1 = i1, i0
            # include endpoint
            seq = acc[i0 : i1 + 1].astype(float, copy=True)
            if seq.size >= 2:
                if label in ref_sequence_filter_list or (len(ref_sequence_filter_list) == 0):
                    sequences.append(seq)
                    labels.append(label)
                    scale_factors.append(settings.ref_sequence_filter_and_scale[label]) # load the scale factor for this sequence
    if not sequences:
        raise ValueError("No reference sequences parsed from JSON; check your export format.")
    return sequences, labels, scale_factors


# ----------------------------
# DTW worker (meant for separate process)
# ----------------------------

@dataclass
class DTWSettings:
    min_match_len: int
    threshold: float
    bufsize: int
    blackout_sec: float 
    # number of processes to use *inside* the DTW process
    nprocs: int = max(1, (os.cpu_count() or 2) - 1)  # leave one core for UI/IO


# Globals used by worker processes (set in _pool_initializer)
_G_REF_SEQS = None
_G_MINLEN = None

class DTWWorker:
    def __init__(self, ref_sequences: list[np.ndarray], ref_labels: list[str], scale_factors: list[float],settings: DTWSettings):
        self.ref_sequences = [np.asarray(r, dtype=float) for r in ref_sequences]
        self.ref_labels = list(ref_labels)
        self.scale_factors = list(scale_factors)
        self.s = settings

        self.t_buf: deque[float] = deque(maxlen=self.s.bufsize)
        self.y_buf: deque[float] = deque(maxlen=self.s.bufsize)
        self._last_end_time: float | None = None

        self._pool = None  # created in run()

    # ---------------- Pool helpers ----------------
    @staticmethod
    def _pool_initializer(ref_sequences: list[np.ndarray], min_match_len: int):
        """Runs once per pool worker: stash refs to avoid re-pickling per task."""
        global _G_REF_SEQS, _G_MINLEN
        _G_REF_SEQS = [np.asarray(r, dtype=float) for r in ref_sequences]
        _G_MINLEN = int(min_match_len)

    @staticmethod
    def _pool_task(args):
        """One task per reference snippet. Returns (idx, start, end, dist) or None."""
        idx, series_np = args
        q = _G_REF_SEQS[idx]
        sa = SubsequenceAlignment(q, series_np, use_c=True)
        bm = next(sa.best_matches_fast(minlength=_G_MINLEN), None)
        if bm is None:
            return None
        s, e = bm.segment
        return (idx, int(s), int(e), float(bm.value))

    # ---------------- Core compute ----------------
    def _compute(self):
        if len(self.y_buf) < self.s.min_match_len:
            return None

        series_np = np.fromiter(self.y_buf, dtype=float)
        times_np = np.fromiter(self.t_buf, dtype=float)

        # Optional trace
        try:
            print(f"Computing DTW at: {times_np.max():.3f}")
        except ValueError:
            # times_np can be empty if buffers are tiny; just skip printing
            pass

        # Skip any data we've already labeled to avoid duplicate matches
        if (self._last_end_time is not None) and (self._last_end_time in times_np):
            new_mask = np.where(times_np > self._last_end_time)
            series_np = series_np[new_mask]
            times_np = times_np[new_mask]
            if len(series_np) < self.s.min_match_len:
                return None

        nref = len(self.ref_sequences)
        if nref == 0:
            return None

        best_label, best_start, best_end, best_dist = None, None, None, math.inf

        # Heuristic chunksize to reduce IPC overhead
        chunksize = max(1, nref // (4 * max(1, self.s.nprocs)))

        for res in self._pool.imap_unordered(
            self._pool_task,
            ((i, series_np) for i in range(nref)),
            chunksize=chunksize,
        ):
            if res is None:
                continue
            idx, s, e, dist = res
            sf = self.scale_factors[idx]
            if (dist/sf) < best_dist:
                best_label = self.ref_labels[idx]
                best_start, best_end, best_dist = s, e, (dist/sf)

        if best_label is None:
            return None

        if (best_dist <= self.s.threshold) and (times_np[best_end]< (times_np.max()-self.s.blackout_sec)):
            t0 = float(times_np[best_start])
            t1 = float(times_np[best_end])
            self._last_end_time = t1
            return (best_label, t0, t1, best_dist)

        return None

    # ---------------- Main loop ----------------
    def run(self, in_q, out_q, stop_evt, poll_timeout=0.05):
        """
        Drain (t, y) chunks from in_q, extend ring buffers, compute once per batch,
        and push the best match (if any) to out_q as (label, t_start, t_end, dist).
        """
        import queue as pyqueue
        import multiprocessing as mp

        # Build the intra-worker process pool once (grandchildren of the UI process)
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=max(1, self.s.nprocs),
            initializer=self._pool_initializer,
            initargs=(self.ref_sequences, self.s.min_match_len),
            maxtasksperchild=256,   # helps with long runs / leaks
        )

        try:
            while not stop_evt.is_set():
                chunks_t = []
                chunks_y = []

                # Block briefly for the first item (avoid busy-wait when idle)
                try:
                    item = in_q.get(timeout=poll_timeout)
                    if item is not None:
                        t, y = item
                        chunks_t.append(np.asarray(t, dtype=float))
                        chunks_y.append(np.asarray(y, dtype=float))
                except pyqueue.Empty:
                    # Nothing arrived this tick
                    continue

                # Drain the rest without blocking
                while True:
                    try:
                        item = in_q.get_nowait()
                    except pyqueue.Empty:
                        break
                    if item is not None:
                        t, y = item
                        chunks_t.append(np.asarray(t, dtype=float))
                        chunks_y.append(np.asarray(y, dtype=float))

                # Batch-extend ring buffers once
                if chunks_t:
                    tcat = np.concatenate(chunks_t)
                    ycat = np.concatenate(chunks_y)
                    self.t_buf.extend(tcat.tolist())
                    self.y_buf.extend(ycat.tolist())

                # Single compute per batch
                result = self._compute()
                if result is not None:
                    out_q.put(result)

        finally:
            # Robust teardown
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass



# ----------------------------
# Phyphox control & reader
# ----------------------------

def stop_experiment(phone_url: str) -> None:
    try:
        resp = requests.get(url=f"{phone_url}/control?cmd=stop").json()
        print("Stop Response:", resp)
    except Exception:
        pass


def clear_experiment(phone_url: str) -> None:
    try:
        resp = requests.get(url=f"{phone_url}/control?cmd=clear").json()
        print("Clear Response:", resp)
    except Exception as e:
        print("Clear failed:", e)


def start_experiment(phone_url: str) -> None:
    try:
        resp = requests.get(url=f"{phone_url}/control?cmd=start").json()
        print("Start Response:", resp)
    except Exception as e:
        print("Start failed:", e)


def reader_thread(settings: Settings, plot_q: Queue, dtw_in_q: Queue, stop_evt):
    last_update_time = 0.0
    phone_url = settings.phone_url
    chans = settings.data_channels
    time_ch = settings.time_channel

    while not stop_evt.is_set():
        try:
            url = phone_url + "/get?" + ("&".join([c + "=" + str(last_update_time) + "|" + time_ch for c in chans]))
            data = requests.get(url=url, timeout=2.0).json()
            buf = data.get("buffer", {})
            series = [buf[tag]["buffer"] for tag in chans]
            times = series[-1]
            if not times:
                time.sleep(0.01)
                continue
            m = min(len(x) for x in series)
            series = [np.asarray(x[:m], dtype=float) for x in series]
            times = np.asarray(times[:m], dtype=float)
            last_update_time = float(times[-1])
            plot_q.put((times, series))
            dtw_y = series[settings.dtw_channel_index]
            put_latest(dtw_in_q, (times, dtw_y))
        except Exception:
            if stop_evt.is_set():
                break
            time.sleep(0.05)


# ----------------------------
# UI / fastplotlib
# ----------------------------

def run_ui(settings: Settings, ref_sequences: list[np.ndarray], ref_labels: list[str], scale_factors: list[float]):
    import multiprocessing as mp
    # UI-only env + imports so child processes don’t initialize graphics stacks.
    # disable the qt accessibility check when querying window state
    os.environ["QT_ACCESSIBILITY"] = "0"   # Qt 5/6
    os.environ["NO_AT_BRIDGE"] = "1"       # also stops GTK/AT-SPI autostart
    import fastplotlib as fpl

    # --- DTW process ---
    ctx = mp.get_context("spawn")
    dtw_in_q: mp.Queue = ctx.Queue(maxsize=10000)   # latest-only
    dtw_out_q: mp.Queue = ctx.Queue(maxsize=64)
    stop_evt = ctx.Event()

    worker = DTWWorker(ref_sequences, ref_labels, scale_factors, DTWSettings(settings.min_match_len, settings.dtw_threshold, settings.dtw_buffer_size, blackout_sec=settings.dtw_detection_blackout_sec))
    proc = ctx.Process(target=worker.run, args=(dtw_in_q, dtw_out_q, stop_evt), daemon=False) # daemon False because we spawn child processes inside a child process - can't do with daemon
    proc.start()

    # --- Initialize Plot Queue thread ---
    plot_q: Queue = Queue(maxsize=10000)

    # --- Buffers similar to the original ---
    PP_CHANNELS = list(settings.data_channels)  # e.g., ["acc", "acc_time"]
    MAX_BUFFER_SIZE = settings.max_buffer_size

    x_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    y_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    z_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]

    state = SimpleNamespace(
        last_recorded_step_type=None,
        last_recorded_step_start=None,
        last_recorded_step_end=None,
    )

    # --- Figure layout identical to original ---
    N = len(PP_CHANNELS) - 1
    canvas_w, canvas_h = 1800, 560

    left_width_frac = 0.66
    right_x0 = left_width_frac
    left_height_frac = 1.0 / max(1, N)

    rects = []
    for i in range(N):
        rects.append(
            (
                0.0,
                1.0 - (i + 1) * left_height_frac,
                left_width_frac,
                left_height_frac,
            )
        )
    rects.append((right_x0, 0.0, 1.0 - right_x0, 1.0))

    figure = fpl.Figure(
        rects=rects,
        size=(canvas_w, canvas_h),
        names=[f"{PP_CHANNELS[i]}" for i in range(N)] + ["Detections"],
    )
    
    # rolling buffer + text on bottom-most subplot
    fps_times = deque(maxlen=10)

    # Colormap & lines like original
    cmap = "tab10"
    for i, subplot in enumerate(figure[:-1]):
        initial = np.column_stack((np.zeros(MAX_BUFFER_SIZE), np.zeros(MAX_BUFFER_SIZE)))
        colors = np.zeros(MAX_BUFFER_SIZE, dtype=int)
        subplot.add_line(initial, cmap=cmap, cmap_transform=colors, name="line")
        subplot.title = PP_CHANNELS[i]
        if i == len(figure[:-1]) - 1:
            fps_text = subplot.add_text(
                "",                         # start empty
                anchor="bottom_right"       # pin the text's bottom-right corner
            )

    # --- update_data matches the original behavior ---
    def update_data():
        # drain plot queue
        last_plot_time = 0
        times = []
        while True:
            try:
                times, series = plot_q.get_nowait()
            except Empty:
                break
            # For each *data* channel except the time channel, extend buffers
            # In your original, PP_CHANNELS = ["acc", "acc_time"], so j=0 is the series, j=1 is time
            for j in range(len(PP_CHANNELS) - 1):
                vals = series[j]
                x_data[j].extend(times.tolist())
                y_data[j].extend(vals.tolist())

        new_plot_time = x_data[0][-1]

        # update plots
        if new_plot_time and (new_plot_time > last_plot_time):
            for j, subplot in enumerate(figure[:-1]):
                new_data = np.empty((MAX_BUFFER_SIZE, 3), np.float32)
                xs = np.array(x_data[j][-MAX_BUFFER_SIZE:])
                ys = np.array(y_data[j][-MAX_BUFFER_SIZE:])
                new_data[:, 0] = xs
                new_data[:, 1] = ys
                new_data[:, 2] = np.array(z_data[j][-MAX_BUFFER_SIZE:])

                # Point colors (string list) with highlighted span if we have a detection
                lbl = ["blue"] * len(xs)
                if state.last_recorded_step_type and len(xs) > 0:
                    mask = (state.last_recorded_step_start is not None) and (state.last_recorded_step_end is not None)
                    if mask:
                        step_idxs = np.where((state.last_recorded_step_start <= xs) & (xs <= state.last_recorded_step_end))[0]
                        if step_idxs.size >= 2:
                            min_idx = int(step_idxs.min())
                            max_idx = int(step_idxs.max())
                            step_color = (
                                "lime"
                                if state.last_recorded_step_type == "Cleese"
                                else "orange"
                                if state.last_recorded_step_type == "Palin"
                                else "white"
                            )
                            for k in range(min_idx, max_idx):
                                lbl[k] = step_color

                subplot["line"].data = new_data
                subplot["line"].colors = lbl
                # keep the original behavior
                subplot.auto_scale(maintain_aspect=False)

            # update rolling-average FPS
            now = time.time()
            fps_times.append(now)
            if len(fps_times) > 1:
                elapsed = fps_times[-1] - fps_times[0]
                fps = (len(fps_times) - 1) / elapsed if elapsed > 0 else 0.0
                fps_text.text = f"{fps:.1f} FPS"
                fps_text.offset = np.array([xs[-1],0,0])

    figure.add_animations(update_data)

    # --- Right "Detections" panel identical to original ---
    text_placeholder = " " * 128
    text_subplot = figure[-1]

    text_gfx = text_subplot.add_text(
        text=text_placeholder,
        font_size=32,
        face_color="lightblue",
        screen_space=True,
        offset=(0, 0, 0),
        anchor="top-left",
    )

    # Invisible scatter to set extents (like original)
    text_subplot.add_scatter(data=np.array([[0, 0, 0], [800, -275, 0]]), alpha=0.0)

    def update_text():
        # drain DTW detections
        got = False
        while True:
            try:
                step_class, start_time, end_time, dist = dtw_out_q.get_nowait()
                got = True
            except Empty:
                break
        if not got:
            return
        state.last_recorded_step_start = start_time
        state.last_recorded_step_end = end_time
        state.last_recorded_step_type = step_class
        color = "lime" if step_class == "Cleese" else "orange" if step_class == "Palin" else "gray"
        text_gfx.text = (
            f"Last Step: {start_time:.3f} --- {end_time:.3f}\nClassification: {step_class}\nDTW Distance: {dist:.3f}"
        ).ljust(len(text_placeholder))
        text_gfx.face_color = color

    figure.add_animations(update_text)
    text_subplot.axes.visible = False
    text_subplot.auto_scale()

    # --- Detect window close and shutdown everything ---
    shutting_down = False

    def _teardown():
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        # 1) ask DTW process to stop
        stop_evt.set()
        try:
            # Nudge the DTW worker off any blocking get()
            dtw_in_q.put_nowait(None)
        except Exception:
            pass
        try:
            stop_experiment(settings.phone_url)
        except Exception:
            pass
        # 2) give DTW process time to exit cleanly (it will close/join its Pool)
        try:
            proc.join(timeout=3.0)
        except Exception:
            pass
        # 3) if it's still alive, escalate
        if proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=1.0)
            except Exception:
                pass
        if proc.is_alive() and hasattr(proc, "kill"):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            fpl.loop.stop()
        except Exception:
            pass

    # atexit safety net
    atexit.register(_teardown)

    # SIGINT -> teardown
    def _sigint(_sig, _frm):
        _teardown()

    signal.signal(signal.SIGINT, _sigint)


    figure.show()
    clear_experiment(settings.phone_url)
    # --- Start Reader thread ---
    r = Thread(target=reader_thread, args=(settings, plot_q, dtw_in_q, stop_evt), daemon=True) 
    r.start()
    start_experiment(settings.phone_url)

    try:
        fpl.loop.run()
    finally:
        _teardown()


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Live plot with process-based DTW and fastplotlib")
    parser.add_argument("--reference-file", type=str, required=True, help="Path to Label Studio export JSON")
    parser.add_argument("--plot-window", type=float, default=10.0)
    parser.add_argument("--dtw-buffer", type=int, default=1000)
    parser.add_argument("--dtw-threshold", type=float, default=9.0)
    parser.add_argument("--min-match-time", type=float, default=0.75, help="seconds")
    parser.add_argument("--approximate-timestep", type=float, default=0.01, help="seconds")
    parser.add_argument("--dtw-channel", type=int, default=0, help="index of series to feed to DTW (0=first)")
    parser.add_argument("--phyphox", type=str, default=os.getenv("PHYPHOX_ADDRESS", "127.0.0.1:8080"), help="host:port")
    args = parser.parse_args()
    return args


def main():
    load_dotenv()
    settings = Settings()

    reference_files = [
        #'label_studio_export/project-2-at-2025-07-26-03-40-4a2f6183.json',
        #'label_studio_export/forward_walk_export.json',
        'label_studio_export/label_studio_export.json' # contains all steps
        ]

    ref_sequences = []
    ref_labels = []
    ref_scale_factors = []
    for reference_file in reference_files:
        file_sequences, file_labels, file_scale_factors = load_ref_sequences(reference_file,settings)
        ref_sequences.extend(file_sequences)
        ref_labels.extend(file_labels)
        ref_scale_factors.extend(file_scale_factors)
    print(f"Loaded {len(ref_sequences)} reference snippets; minlen={settings.min_match_len} samples")

    run_ui(settings, ref_sequences, ref_labels, ref_scale_factors)


if __name__ == "__main__":
    # Windows-safe multiprocessing entry
    main()
