import matplotlib
matplotlib.use("TkAgg")
from units_to_use import units_to_use, channel_map, custom_correction
import numpy as np
import scipy.stats
import scipy.io
import scipy.signal
import cv2
import os
from suss.io import read_pickle
import matplotlib.pyplot as plt
# from segmentation.audio_segment import merge_segments
from preprocess.sound.stimuli import merge_segments
from soundsig.sound import spectrogram, plot_spectrogram
from soundsig.signal import highpass_filter, bandpass_filter
from matplotlib.widgets import Button, SpanSelector, RadioButtons, Slider
from matplotlib.patches import Rectangle
from itertools import cycle
from functools import partial
import time
import quantities as pq
import config


plt.rcParams['keymap.fullscreen'] = ''
plt.rcParams['keymap.back'] = ''
plt.rcParams['keymap.forward'] = ''
plt.rcParams['keymap.pan'] = ''
plt.rcParams['keymap.zoom'] = ''
plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.quit'] = ''
plt.rcParams['keymap.grid'] = ''
plt.rcParams['keymap.yscale'] = ''
plt.rcParams['keymap.xscale'] = ''
plt.rcParams['keymap.all_axes'] = ''
plt.rcParams['toolbar'] = 'None'


# class ClickClickInterval(object):
#     def __init__(self, ax, callback, direction="horizontal", rectprops=None):
#         if direction not in ("horizontal", "vertical"):
#             raise ValueError("direction must be 'horizontal' or 'vertical'")
        
#         if rectprops is None:
#             rectprops = {}
#         self.ax = ax
#         self.callback = callback
#         self.direction = direction
#         self.last_point = None
#         self.interval = None
#         self.awaiting_second_point = False
#         self.rect = Rectangle(
#             (0, 0),
#             0, 
#             1, 
#             **rectprops)
#         self.rect.set_visible(False)
#         self.ax.add_patch(self.rect)
#         self.canvas = self.ax.figure.canvas
#         self.canvas.mpl_connect("button_press_event", self.on_click)
        
#     def clear(self):
#         self.last_point = None
#         self.awaiting_second_point = False
#         self.rect.set_visible(False)
        
#     def on_click(self, event):
#         if direction == "horizontal":
#             x = event.xdata if direction == "horizontal" else event.ydata
        
#         if not self.awaiting_second_point:
#             self.clear()
#             self.last_point = x
#             self.awaiting_second_point = True
#             self.rect.set_bounds(x, self.ax.get_ylim()[0], 0, self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
#             self.rect.set_visible(True)

#         else:
#             self.interval = sorted([self.last_point, x])
#             self.callback(*self.interval)
#             self.rect.set_bounds(
#                 self.interval[0], 
#                 self.ax.get_ylim()[0],
#                 self.interval[1] - self.interval[0], 
#                 self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
#             )
#             self.rect.set_visible(True)
#             self.awaiting_second_point = False
#             self.last_point = None
            

class App(object):
    MODES = ["Bout", "Motif", "Syllable", "Note"]
    COLORS = cycle(["#1DD135", "#0681C8", "#C41DD2", "#F19200", "#5E06E7", "#E70808", "#3052D1", "#DADD2E"])

    def __init__(self, events, loader, savepath=None, loadpath=None, figsize=(8, 8)):
        self.events = events
        self.fs = loader.get_signal_sampling_rate("mic")

        if savepath is None:
            self.savepath = loadpath or "temp_syllable_labeling.npy"
        
        if loadpath is not None and os.path.exists(loadpath):
            loaded_data = np.load(loadpath, allow_pickle=True)[()]
            print("Loaded", loadpath)
            self.labels = loaded_data["labels"]
            self.assigned_colors = loaded_data["assigned_colors"]
            print("Loaded colors", self.assigned_colors)

            self.templates = loaded_data["templates"]
            self.i = loaded_data.get("i", 0)
        
        else:
            # Mapping of mode -> mapping of label to list of syllables
            self.labels = {}
            self.i = 0
            self.assigned_colors = {}
            self.templates = {
                "Syllable": {},
                "Note": {},
            }
            for mode in self.MODES:
                self.labels[mode] = {}
                self.assigned_colors[mode] = {}

        self.channel = channel_map.get(SITE, 0)
        self._event_times = np.array([e["start_time"].item() for e in self.events])
        self.loader = loader
        self.figsize = figsize
        self.keypress_cid = None
        self.last_spec = None
        self.last_spec_time = None
        self.noise_level = 65
        self.OFFSET_BEFORE = 1.0
        self.OFFSET_AFTER = 2.0
        self.MAX_TEMPLATE = 250   # 0.5 s
        self.mode = self.MODES[2]
        self.highlighted = None
        self.selected_rect = None
        self.fig = None
        
        self.label_buttons = []
        self.button_callbacks = []
        
        self._action_stack = []
#         self._redo_stack = []

    def save(self, event=None):
        to_save = {
            "templates": self.templates,
            "labels": self.labels,
            "assigned_colors": self.assigned_colors,
            "i": self.i
        }
        np.save(self.savepath, to_save)
        self.debug("Saving successful")
        
    def _get_range(self, event):
        return [event["start_time"].item(), (event["start_time"] + event["duration"]).item()]
    
    def debug(self, *args):
        try:
            if hasattr(self, "debug_text"):
                if len(args) > 1:
                    self.debug_text.set_text("{}".format(args))
                else:
                    self.debug_text.set_text(str(args[0]))
        except:
            pass  # hacky way to not let debug accidentally block anything

    @property
    def current_event(self):
        print(self.i, len(self.events))
        return self.events[self.i]
    
    def jump_to(self, i):
        self.i = i

        self.reset_selectors()

        self.matcher_ax.clear()
        self.matcher_ax.axis("off")
        self.draw()
#         self.run()

    def next(self, event):
        self.jump_to(min(len(self.events) - 1, self.i + 1))

    def prev(self, event):
        self.jump_to(max(0, self.i - 1))
        
    def on_select_time(self, t):
        idx = np.searchsorted(self._event_times, t)
        self.jump_to(idx)

    def on_select_range(self, tmin, tmax):
        # if any of the highlighted region is now out of range, clear it
        if self.highlighted is not None:
            if tmin > self.highlighted[1] or tmax < self.highlighted[0]:
                self.highlighted = None
            
        self.spec_ax.set_xlim(tmin, tmax)
        self.matcher_ax.set_xlim(tmin, tmax)
        self.labels_ax.set_xlim(tmin, tmax)
        
    def on_highlight_syllable(self, tmin, tmax):
        self.highlighted = (tmin, tmax)

    def on_cut(self, cutmin, cutmax):
        # Cut self.current_event up... potentially extending it
        return
        
    def on_delete(self):
        # Delete the self.current_event
        return
    
    def on_assign_label(self, label, event):
        self.assign_label(label)
        
    def get_next_label(self):
        if self.mode in ("Syllable", "Note"):
            if self.labels[self.mode]:
                curr_label = np.max([ord(label) for label in self.labels[self.mode]])
                return chr(curr_label + 1)
            else:
                return "A" if self.mode == "Syllable" else "a"
        elif self.mode == "Bout":
            return "B"
        elif self.mode == "Motif":
            return "M"
        
    def on_key_press(self, event):
        self.debug("Registering key press {}".format(event.key))
        
        if self.selected_rect is not None and event.key in ("delete", "backspace"):
            mode, label, syl = self.selected_rect.syllable_
            events = self.labels[mode][label]
            idx = events.index(syl)
            self._action_stack.append({
                "action": "delete",
                "type": self.mode,
                "label": label,
                "syllable": syl
            })
            events.pop(idx)
            if not events:
                self.labels[mode].pop(label, None)
                self.templates[mode].pop(label, None)
            self.draw()
            return
      
        if event.key == " ":
            self.run_autolabel(None)
            return

        if event.key == "right":
            self.next(None)
            return
            
        if event.key == "left":
            self.prev(None)
            return
            
        if event.key == "ctrl+z":
            return self.undo()
    
        if event.key == "ctrl+n" or event.key.lower() == self.get_next_label().lower():
            self.assign_label(self.get_next_label())
            return

        if self.mode in ("Syllable", "Note"):
            if event.key.lower() in set([l.lower() for l in self.labels[self.mode]]):
                if self.mode == "Syllable":
                    self.assign_label(event.key.upper())
                elif self.mode == "Note":
                    self.assign_label(event.key.lower())
    
    def on_label_pick(self, event):
        if isinstance(event.artist, Rectangle):
            if self.selected_rect is not None:
                self.selected_rect.set_alpha(0.5)
            self.selected_rect = event.artist
            self.selected_rect.set_alpha(1.0)
            self.debug("Delete selected syllable with 'delete' key")

    def undo(self):
        if not self._action_stack:
            return
        
        last_action = self._action_stack.pop()
        self.debug(last_action)
        if last_action["action"] == "create":
            events = self.labels[last_action["type"]][last_action["label"]]
            events.pop()
#             self._redo_stack.append({
#                 "type": last_action["type"],
#                 "label": last_action["label"],
#                 "event": events.pop()
#             })
            if not events:
                self.labels[last_action["type"]].pop(last_action["label"], None)
                self.templates[last_action["type"]].pop(last_action["label"], None)
        self.draw()
        
    def redo(self):
        if not self._redo_stack:
            return
        
        raise Exception("Haven't implemented redo yet")
                    
    def assign_label(self, label):
        if self.highlighted is None:
            return
        
        if label not in self.labels[self.mode]:
            self.labels[self.mode][label] = []
        
        if label not in self.assigned_colors[self.mode]:
            self.assigned_colors[self.mode][label] = next(self.COLORS)
            
        self.labels[self.mode][label].append({
            "t_start": self.highlighted[0] + self.current_event["start_time"].item(),
            "t_stop": self.highlighted[1] + self.current_event["start_time"].item(),
            "label": label
        })
        self.labels[self.mode][label] = sorted(self.labels[self.mode][label], key=lambda e: e["t_start"])
        
        self._action_stack.append({
            "action": "create",
            "type": self.mode,
            "label": label
        })
        
        self.debug("Marked {} {} at {:.3f}".format(
            self.mode,
            label,
            self.labels[self.mode][label][-1]["t_start"]))
        
        if self.mode in ("Syllable", "Note"):
            t_slice = np.where((self.t_spec > self.highlighted[0]) & (self.t_spec < self.highlighted[1]))[0]
            new_spec = self.spec[:, t_slice]
            
            # Compute template
            if label not in self.templates[self.mode]:
                self.templates[self.mode][label] = (1, new_spec)
            else:
                n, curr_template = self.templates[self.mode][label]
                self.templates[self.mode][label] = min(n + 1, 20), self._update_template(n, curr_template, new_spec)

        self.draw()
        
    def on_refresh(self, event=None):
        # refresh templates
        if self.mode not in ("Syllable", "Note"):
            return
        
        self.debug("Refreshing templates...")
        for label in sorted(self.labels[self.mode]):
            self.debug("Refreshing templates {}".format(label))
            syls = self.labels[self.mode][label]
            specs = []
            durs = []
            for i, syl in enumerate(syls):
                self.debug("Loading templates {} {}/{}".format(label, i, len(syls)))

                data = self.loader.time_slice(syl["t_start"], syl["t_stop"], signal="mic").data[:, self.channel]
                data = bandpass_filter(data - np.mean(data), self.fs, 500, 10000)
                t_spec, f_spec, spec, _ = spectrogram(data, self.fs, 1000, 50, min_freq=250, max_freq=10000, cmplx=False)
                specs.append(spec)
                durs.append(len(t_spec))
            
            width = int(round(np.median(durs)))
            full_spec = np.zeros((spec.shape[0], width))
            
            for i, spec in enumerate(sorted(specs, key=lambda s: s.shape[1], reverse=True)):
                
                if np.abs(width - spec.shape[1]) % 2 == 1:
                    d = np.random.choice([0, 1])
                else:
                    d = 0
                    
                if spec.shape[1] < width:
                    start = d + (width - spec.shape[1]) // 2
                    full_spec[:, start:start + spec.shape[1]] *= (i / (i + 1))
                    full_spec[:, start:start + spec.shape[1]] += (1 / (i + 1)) * spec
                    
                elif spec.shape[1] >= width:
                    start = d + (spec.shape[1] - width) // 2
                    self.debug(full_spec)
                    full_spec *= (i / (i + 1))
                    full_spec += (1 / (i + 1)) * spec[:, start:start + width]
                    
            self.templates[self.mode][label] = (len(specs), full_spec)
            
        self.debug("Templates refreshed...")

        self.draw()
        
    def setup_labels(self):
        """Draw labels above the spectrograms"""
        self.labels_ax.clear()
        self.labels_ax.axis("off")
        self.labels_ax.set_xlim(*self.spec_ax.get_xlim())

        for label in sorted(self.labels[self.mode]):
            start = time.time()
            n = 0
            
            syls = self.labels[self.mode][label]
            idx = np.searchsorted([s["t_start"] for s in syls], self.current_event["start_time"].item() - self.OFFSET_BEFORE)
            for syl in syls[idx:]:
                n += 1
                t0 = syl["t_start"] - self.current_event["start_time"].item()
                duration = syl["t_stop"] - syl["t_start"]

                if syl["t_start"] > (self.current_event["start_time"] + self.current_event["duration"]).item() + self.OFFSET_AFTER:
                    print("breaking after {}".format(n))
                    break

                rect = Rectangle(
                    (t0 + 0.005, 0), 
                    duration - 0.01, 
                    1, 
                    picker=True,
                    edgecolor="Black", 
                    facecolor=self.assigned_colors[self.mode][label],
                    alpha=0.5)
                rect.syllable_ = (self.mode, label, syl)
                self.labels_ax.text(t0 + (duration / 2), 1.1, label, horizontalalignment="center")
                self.labels_ax.add_patch(rect)
            print("Finsihed {} in {:.2f}".format(label, time.time() - start))

        self.fig.canvas.draw_idle()
        
    def _update_template(self, n, template, addition):
        t = template.shape[1]
        a = addition.shape[1]
        
        if np.abs(t - a) % 2 == 1:
            d = np.random.choice([0, 1])
        else:
            d = 0
            
        if t > a:
            spot = d + int(round((t - a) / 2))
            result = template
            result[:, spot:spot + a] *= n / (n + 1)
            result[:, spot:spot + a] += addition / (n + 1)
        elif a > t:
            spot = d + int(round((a - t) / 2))
            result = addition
            result[:, spot:spot + t] *= 1 / (n + 1)
            result[:, spot:spot + t] += template * n / (n + 1)
        else:
            result = (addition / (n + 1)) + (template * n / (n + 1))
        
        if result.shape[1] > self.MAX_TEMPLATE:
            spot = int(round((result.shape[1] - self.MAX_TEMPLATE) / 2))
            result = result[:, spot:spot + self.MAX_TEMPLATE]
        
        # make the saved template leak 1 sample from the ends
        return result
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.draw()

    def select_channel(self, channel):
        self.channel = int(channel)
        self.draw()
    
    def select_mode(self, mode):
        self.mode = mode
        self.draw()
        
    def reset_selectors(self):
        self.highlighted = None

    def run_autolabel(self, event):
        self.matcher_ax.clear()
        self.matcher_ax.axis("off")
        if self.mode not in ("Syllable", "Note"):
            return
        
        self.debug("Running autolabel...")
        
        occupied_intervals = []
        for label in self.templates[self.mode]:
            occupied_intervals += [
                (syl["t_start"], syl["t_stop"]) for syl in self.labels[self.mode][label]
            ]

        finished = []
        for label in sorted(self.templates[self.mode],
                            key=lambda l: self.templates[self.mode][l][1].shape[1],
                            reverse=True):
            finished.append(label)
            self.debug(finished)
            syls = self.labels[self.mode][label]
            current_syls = [
                syl for syl in syls
                if (
                    self.current_event["start_time"].item() - self.OFFSET_BEFORE < 
                    syl["t_start"] < 
                    self.current_event["start_time"].item() + self.current_event["duration"].item() + self.OFFSET_AFTER
                )
            ]
            
            durations = []
            for syl in syls:
                t0 = syl["t_start"] - self.current_event["start_time"].item()
                t1 = syl["t_stop"] - self.current_event["start_time"].item()
                durations.append(t1 - t0)
            
            for syl in current_syls:
                t0 = syl["t_start"] - self.current_event["start_time"].item()
                t1 = syl["t_stop"] - self.current_event["start_time"].item()
                t_slice = np.where((self.t_spec > t0) & (self.t_spec < t1))[0]
                new_spec = self.spec[:, t_slice]
                n, curr_template = self.templates[self.mode][label]
                self.templates[self.mode][label] = min(n + 1, 20), self._update_template(n, curr_template, new_spec)
            
            mean_duration = np.median(durations) / 2
            mean_width = int(round(mean_duration * 1000))
            _, template = self.templates[self.mode][label]

            convolved = cv2.matchTemplate(
                self.spec.astype(np.float32),
                template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )[0]

            t_conv = self.t_spec[template.shape[1] // 2 - 1:-template.shape[1] // 2]
            self.debug(convolved.shape, t_conv.shape)
            
            peaks = scipy.signal.find_peaks_cwt(convolved, np.arange(10, 50, 5))
            peaks = peaks[convolved[peaks] > 0.6]
            self.matcher_ax.plot(t_conv, convolved, color=self.assigned_colors[self.mode][label], linewidth=0.5)
            self.matcher_ax.plot(t_conv[peaks], convolved[peaks], 
                linestyle="none", marker="*", color=self.assigned_colors[self.mode][label])
            count = 0
            for peak in peaks:
                t_peak = t_conv[peak]
                self.debug(occupied_intervals)
                self.debug(
                    "{:.2f}, {:.2f}".format(*occupied_intervals[0]),
                    "{:.2f}".format(t_peak + self.current_event["start_time"].item())
                )
                if np.any([t0 < (t_peak + self.current_event["start_time"].item()) < t1 for t0, t1 in occupied_intervals]):
                    continue
                if (t_peak - mean_duration) < self.t_spec[0]:
                    continue
                if (t_peak + mean_duration) > self.t_spec[-1]:
                    continue
                    
                count += 1
                if (peak - mean_width) < 0 or (peak + mean_width) >= len(t_conv):
                    continue
                new_syl = {
                    "t_start": self.current_event["start_time"].item() + t_conv[peak-mean_width],
                    "t_stop": self.current_event["start_time"].item() + t_conv[peak+mean_width],
                    "label": label
                }
                occupied_intervals.append((new_syl["t_start"], new_syl["t_stop"]))
                self.labels[self.mode][label].append(new_syl)
                self.labels[self.mode][label] = sorted(self.labels[self.mode][label], key=lambda e: e["t_start"])

#             self.debug("Found {} for {}".format(count, label))

        self.draw()
        
    def draw(self):
        begin = time.time()
        if self.selected_rect is not None:
            self.selected_rect.set_alpha(0.5)
            self.selected_rect = None
            self.debug("--")
        print("A: {:.2f}".format(time.time() - begin))
            
        for button, cbid in self.label_buttons:
            button.disconnect_events()
            try:
                button.ax.remove()
            except:
                pass

        t0, t1 = self._get_range(self.current_event)
        pad_after = max(0.5, self.OFFSET_AFTER - (t1 - t0))
        
        spec_changed = self.last_spec_time != (t0, t1, self.channel)
        print("B: {:.2f}".format(time.time() - begin))
        if not spec_changed:
            t_spec, f_spec, spec = self.last_spec
        else:
            if self.current_event["sound_type"] == "playback":
                fs, data = scipy.io.wavfile.read(self.current_event["playback_file"])
                data = data - np.mean(data)
                data = np.concatenate([
                    np.zeros(int(self.OFFSET_BEFORE * fs)),
                    data,
                    np.zeros(int(pad_after * fs))
                ])
            else:
                data = self.loader.time_slice(t0 - self.OFFSET_BEFORE, t1 + pad_after, signal="mic").data[:, self.channel]
                data = bandpass_filter(data - np.mean(data), self.fs, 500, 10000)
                fs = self.fs

            t_spec, f_spec, spec, _ = spectrogram(data, fs, 1000, 50, min_freq=250, max_freq=10000, cmplx=False)
            t_spec -= self.OFFSET_BEFORE

            # Stash these so that they can be scooped up later by assidefgn_label
            self.t_spec = t_spec
            self.spec = spec
            self.last_spec_time = (t0, t1, self.channel)
            self.last_spec = t_spec, f_spec, spec
        print("C: {:.2f}".format(time.time() - begin))

        if spec_changed:
            plot_spectrogram(t_spec, f_spec, spec, dBNoise=self.noise_level, ax=self.spec_scroll_ax, colormap="viridis", colorbar=False)
            self.spec_scroll_ax.set_xlabel("")
            self.spec_scroll_ax.set_xlim(-self.OFFSET_BEFORE, (t1 - t0) + pad_after)
            plot_spectrogram(t_spec, f_spec, spec, dBNoise=self.noise_level, ax=self.spec_ax, colormap="viridis", colorbar=False)
            self.fig.canvas.draw_idle()
        print("D: {:.2f}".format(time.time() - begin))

        if self.spec_ax.get_xlim()[1] > t_spec[-1]:
            self.spec_ax.set_xlim(t_spec[0], t_spec[-1])
        self.matcher_ax.set_xlim(*self.spec_ax.get_xlim())

        print("E: {:.2f}".format(time.time() - begin))

        self.setup_labels()        
        self.fig.canvas.draw_idle()

        print("F: {:.2f}".format(time.time() - begin))

        self.overall_time_ax.clear()
        self.overall_time_ax.vlines([event["start_time"] for event in self.events], 0, 1, alpha=0.1)
        self.overall_time_ax.vlines([self.current_event["start_time"]], 0, 1, linewidth=2, color="Red")
        self.overall_time_ax.axis("off")

        self.setup_buttons()
        print("G: {:.2f}".format(time.time() - begin))

        self.fig.canvas.draw_idle()

        print("H: {:.2f}".format(time.time() - begin))
        # plt.show()

    def close_fig(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def run(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        
        debug_axis = self.fig.add_axes([ 0.86,  0.93, 0.14, 0.06], xlim=(-1, 1), ylim=(-1, 1))
        self.debug_text = debug_axis.text(0, 0, "", wrap=True, horizontalalignment="center", verticalalignment="center")
        debug_axis.axis("off")
        
        self.overall_time_ax = self.fig.add_axes([0.1, 0.92, 0.65, 0.05], xlim=(0, self.loader.max_time()))
        self.overall_time_ax.axis("off")
        self.overall_time_jump = Slider(
            self.overall_time_ax,
            "",
            0, 
            self.loader.max_time(),
            valinit=0,
            dragging=False)
        self.overall_time_jump.on_changed(self.on_select_time)
        self.spec_scroll_ax = self.fig.add_axes([0, 0.75, 0.85, 0.15], xlim=(-self.OFFSET_BEFORE, self.OFFSET_AFTER))
        self.spec_scroll_ax.set_xlabel("View Range")
        self.slider = SpanSelector(
            self.spec_scroll_ax,
            self.on_select_range,
            "horizontal", 
            0.1,
            useblit=True,
            span_stays=True,
            rectprops={
                "facecolor": "none",
                "edgecolor": "Red",
                "alpha": 0.9
            }
        )
        
        self.labels_ax = self.fig.add_axes([ 0, 0.6, 0.85, 0.02])
        self.fig.canvas.mpl_connect("pick_event", self.on_label_pick)
        self.labels_ax.axis("off")
        
        self.matcher_ax = self.fig.add_axes([ 0, 0.65, 0.85, 0.05])
        self.matcher_ax.axis("off")
        
        self.spec_ax = self.fig.add_axes([ 0, 0.3, 0.85, 0.3])
        self.syllable_highlighter = SpanSelector(
            self.spec_ax,
            self.on_highlight_syllable,
            "horizontal",
            0.01,
            useblit=True,
            span_stays=True,
            rectprops={
                "edgecolor": "Green",
                "alpha": 0.25,
                "facecolor": "White",
                "linewidth": 2
            }
        )
        
        noise_ax = self.fig.add_axes([ 0.86,  0.5, 0.09, 0.02], facecolor="White")
        noise_ax.set_title("dBNoise")
        self.noise_slider = Slider(noise_ax, "", 10, 100, valinit=self.noise_level, dragging=False, valstep=5)
        self.noise_slider.on_changed(self.set_noise_level)

        save_ax = self.fig.add_axes([ 0.86,  0.55, 0.13, 0.05], facecolor="White")
        self.save_btn = Button(save_ax, 'Save')
        self.save_btn.on_clicked(self.save)

        autolabel_ax = self.fig.add_axes([ 0.86,  0.61, 0.13, 0.05], facecolor="White")
        self.autolabel_btn = Button(autolabel_ax, 'Autolabel')
        self.autolabel_btn.on_clicked(self.run_autolabel)
    
        refresh_ax = self.fig.add_axes([ 0.86,  0.67, 0.13, 0.05], facecolor="White")
        self.refresh_btn = Button(refresh_ax, 'Refresh\nTemplates')
        self.refresh_btn.on_clicked(self.on_refresh)

        channel_ax = self.fig.add_axes([ 0.86,  0.75, 0.13, 0.08], facecolor="White")
        channel_ax.set_title("Channel")
        self.channel_radio = RadioButtons(channel_ax, (0, 1), active=self.channel)
        self.channel_radio.on_clicked(self.select_channel)

        mode_ax = self.fig.add_axes([ 0.86,  0.3, 0.13, 0.15], facecolor="White")
        mode_ax.set_title("Mode")
        self.mode_radio = RadioButtons(mode_ax, self.MODES, active=2)
        self.mode_radio.on_clicked(self.select_mode)

        button_next_ax = self.fig.add_axes([0.76, 0.92, 0.09, 0.05])
        self.bnext = Button(button_next_ax, 'Next')
        self.bnext.on_clicked(self.next)
        self.bnext.label.set_fontweight("bold")
        
        button_prev_ax = self.fig.add_axes([0, 0.92, 0.09, 0.05])
        self.bprev = Button(button_prev_ax, 'Prev')
        self.bprev.on_clicked(self.prev)
        self.bprev.label.set_fontweight("bold")
        
        self.CLUSTERS_PANEL = [0.02, 0.05, 0.09, 0.15]
        
        self.setup_buttons()
        self.draw()

        plt.show()
    
    def setup_buttons(self):
        begin = time.time()
        if self.keypress_cid:
            self.fig.canvas.mpl_disconnect(self.keypress_cid)

        for button, cbid in self.label_buttons:
            button.disconnect_events()
            try:
                button.ax.remove()
            except:
                pass

        x, y, w, h = self.CLUSTERS_PANEL
        curr_x = x
        self.label_buttons = []
        if self.mode == "Syllable" or self.mode == "Note":
            for label in sorted(self.labels[self.mode]):
                n, template =  self.templates[self.mode][label]
                width = template.shape[1] / 3000
                new_button_ax = self.fig.add_axes([curr_x, y, width, h])
                t_spec = np.arange(template.shape[1])
                f_spec = np.arange(template.shape[0])
                plot_spectrogram(
                    t_spec,
                    f_spec, 
                    template, 
                    ax=new_button_ax, 
                    colormap="Greys",
                    colorbar=False,
                    dBNoise=self.noise_level)
                new_button_ax.axis("off")
                
                new_button = Button(new_button_ax, label)
                new_button.label.set_fontsize(14)
                new_button.label.set_fontweight("bold")
                new_button.label.set_position((0.5, 1.1))
                new_button.label.set_color(self.assigned_colors[self.mode][label])
                cbid = new_button.on_clicked(partial(self.on_assign_label, label))
                self.label_buttons.append((new_button, cbid))

                curr_x += width + 0.02

            new_button_ax = self.fig.add_axes([curr_x, y, w, h])
            new_button = Button(new_button_ax, "+\nNew\nLabel\n(n)")
            cbid = new_button.on_clicked(partial(self.on_assign_label, self.get_next_label()))
            self.label_buttons.append((new_button, cbid))
        else:
            new_button_ax = self.fig.add_axes([curr_x, y, w, h])
            new_button = Button(new_button_ax, "+\nMark\n{}\n({})".format(self.mode, self.get_next_label().lower()))
            cbid = new_button.on_clicked(partial(self.on_assign_label, self.get_next_label()))

            self.label_buttons.append((new_button, cbid))

        self.keypress_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("H: {:.2f}".format(time.time() - begin))


if __name__ == "__main__":
    import sys
    if not len(sys.argv[1:]) in [3, 4]:
        raise Exception("call it with 'python syllable_labeling.py [username] [SITE] ['subject' or 'tutor'] [live or playback]")

    if len(sys.argv[1:]) == 3:
        USER, SITE, VOCALIZER = sys.argv[1:]
        LIVE = "live"
    else:
        USER, SITE, VOCALIZER, LIVE  = sys.argv[1:]

    site_dir = os.path.join(
        ("/auto/fhome/kevin/Projects/songephys/data/birds/"),
        SITE.split("_")[0],
        "sites")
    site = os.path.join(site_dir, SITE)
    if LIVE == "live":
        sound_events = np.load(os.path.join(site, "disjoint_vocal_periods.npy"), allow_pickle=True)[()]
        sound_events = custom_correction(sound_events, SITE)
    elif LIVE == "playback":
        sound_events = np.load(os.path.join(site, "vocal_periods.npy"), allow_pickle=True)[()]["playback"]
        sound_events = custom_correction(sound_events, SITE)
        for event in sound_events:
            event["start_time"] += config.PAD_SONG_BEFORE * pq.s
            event["duration"] -= (config.PAD_SONG_BEFORE + config.PAD_SONG_AFTER) * pq.s

    loader = read_pickle(os.path.join(site, "lazy_data.pkl"))
    save_path = os.path.join(
        site,
        "manual_syllable_labeling",
        "{}_{}{}.npy".format(
            VOCALIZER,
            USER,
            "" if LIVE == "live" else "_{}".format(LIVE)
        )
    )

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    browser = App(
        [
            e for e in sound_events if
            e.get("vocalization_type") == "So" and
            e["sound_type"] == LIVE and
            e.get("vocalizer") == VOCALIZER
        ],
        loader,
        loadpath=save_path
    )
    browser.run()

