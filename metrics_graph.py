import time
import os
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==================== COLOR PALETTE ====================
COLORS = {
    'bg': '#0f172a',
    'panel': '#1e293b',
    'grid': '#334155',
    'text': '#e2e8f0',

    'fps': '#22d3ee',
    'avg': '#facc15',
    'instant': '#fb923c',
    'smooth': '#4ade80',
    'danger': '#f87171',

    'orb_on': '#38bdf8'
}


# ==================== METRICS COLLECTOR ====================
class MetricsCollector:

    STATUS_MAP = {
        'Civitas UB': 1.0,
        'Uncertain': 0.5,
        'Non-Civitas UB': 0.0,
        'Detecting...': -0.1
    }

    BRIGHTNESS_MAP = {
        'Dark': 0,
        'Normal': 1,
        'Bright': 2
    }

    def __init__(self, algorithm='Unknown', max_points=3000, aggregation='mean'):
        self.algorithm = algorithm
        self.aggregation = aggregation
        self.start_time = time.time()

        # time + performance
        self.timestamps = deque(maxlen=max_points)
        self.fps_history = deque(maxlen=max_points)

        # detection
        self.instant_conf = deque(maxlen=max_points)
        self.smooth_conf = deque(maxlen=max_points)
        self.smooth_status = deque(maxlen=max_points)

        # pipeline
        self.brightness = deque(maxlen=max_points)
        self.orb_usage = deque(maxlen=max_points)
        self.track_counts = deque(maxlen=max_points)

        # label
        self.status_labels = deque(maxlen=max_points)

    # ==================== RECORD ====================
    def record(self, fps, people_data, brightness, orb_ran, track_count):
        t = time.time() - self.start_time

        self.timestamps.append(t)
        self.fps_history.append(fps)
        self.track_counts.append(track_count)

        if people_data:
            inst_list = [p['instant_conf'] for p in people_data]
            smooth_list = [p['smooth_conf'] for p in people_data]
            status_list = [self.STATUS_MAP.get(p['status'], -0.1) for p in people_data]

            inst = self._aggregate(inst_list)
            smooth = self._aggregate(smooth_list)
            status = self._aggregate(status_list)

            dominant_status = people_data[np.argmax(smooth_list)]['status']
        else:
            inst, smooth, status = 0, 0, -0.1
            dominant_status = 'Detecting...'

        self.instant_conf.append(inst)
        self.smooth_conf.append(smooth)
        self.smooth_status.append(status)
        self.status_labels.append(dominant_status)

        self.brightness.append(self.BRIGHTNESS_MAP.get(brightness, 1))
        self.orb_usage.append(1 if orb_ran else 0)

    # ==================== AGGREGATION ====================
    def _aggregate(self, values):
        if not values:
            return 0

        if self.aggregation == 'mean':
            return float(np.mean(values))
        elif self.aggregation == 'max':
            return float(np.max(values))
        elif self.aggregation == 'weighted':
            weights = np.linspace(1, 2, len(values))
            return float(np.average(values, weights=weights))
        else:
            return float(np.mean(values))

    # ==================== GRAPH ====================
    def save_graph(self, output_dir='.', prefix=None):
        if len(self.timestamps) < 10:
            print('⚠️ Not enough data.')
            return None

        ts = np.array(self.timestamps)
        fps = np.array(self.fps_history)
        i_c = np.array(self.instant_conf)
        s_c = np.array(self.smooth_conf)
        s_st = np.array(self.smooth_status)
        orb = np.array(self.orb_usage)
        bright = np.array(self.brightness)
        tracks = np.array(self.track_counts)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{prefix or self.algorithm}_FULL_metrics_{stamp}.png"
        path = os.path.join(output_dir, fname)

        fig = plt.figure(figsize=(18, 16), facecolor=COLORS['bg'])
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.7)

        axes = [fig.add_subplot(gs[i]) for i in range(10)]
        _style_axes(axes)

        # 1 FPS over time
        axes[0].plot(ts, fps, color=COLORS['fps'], linewidth=1.8)
        axes[0].axhline(fps.mean(), color=COLORS['avg'], linestyle='--')
        axes[0].fill_between(ts, fps, fps.mean(), color=COLORS['fps'], alpha=0.1)
        axes[0].set_title("FPS Over Time")

        # 2 FPS distribution
        axes[1].hist(fps, bins=30, color=COLORS['fps'], alpha=0.8)
        axes[1].axvline(fps.mean(), color=COLORS['avg'], linestyle='--')
        axes[1].set_title("FPS Distribution")

        # 3 Confidence
        axes[2].plot(ts, i_c, color=COLORS['instant'], alpha=0.4, label='Instant')
        axes[2].plot(ts, s_c, color=COLORS['smooth'], linewidth=2, label='Smoothed')
        axes[2].axhline(0.6, color=COLORS['danger'], linestyle='--')
        axes[2].legend(facecolor=COLORS['panel'], edgecolor='none', labelcolor=COLORS['text'])
        axes[2].set_title("Confidence Over Time")

        # 4 Status timeline
        colors = np.where(s_st == 1.0, '#facc15',
                 np.where(s_st == 0.5, '#fb923c', '#94a3b8'))
        axes[3].scatter(ts, s_st, c=colors, s=6)
        axes[3].set_yticks([1, 0.5, 0])
        axes[3].set_yticklabels(['Civitas', 'Uncertain', 'Non'])
        axes[3].set_title("Status Timeline")

        # 5 ORB usage
        axes[4].plot(ts, orb, color=COLORS['orb_on'])
        axes[4].fill_between(ts, orb, color=COLORS['orb_on'], alpha=0.2)
        axes[4].set_title("ORB Usage")

        # 6 Brightness
        axes[5].plot(ts, bright, color='#a78bfa')
        axes[5].set_yticks([0,1,2])
        axes[5].set_yticklabels(['Dark','Normal','Bright'])
        axes[5].set_title("Brightness")

        # 7 Tracks
        axes[6].plot(ts, tracks, color='#60a5fa')
        axes[6].fill_between(ts, tracks, color='#60a5fa', alpha=0.1)
        axes[6].set_title("Track Count")

        # 8 FPS vs Confidence
        axes[7].scatter(fps, s_c, s=6, alpha=0.6, color='#34d399')
        axes[7].set_title("FPS vs Confidence")

        # 9 ORB impact
        orb_on = s_c[orb == 1]
        orb_off = s_c[orb == 0]

        if len(orb_on) > 0 and len(orb_off) > 0:
            axes[8].boxplot([orb_off, orb_on], patch_artist=True)
            axes[8].set_xticklabels(['ORB OFF', 'ORB ON'])
        else:
            axes[8].text(0.3, 0.5, "Not enough data", color=COLORS['text'])

        axes[8].set_title("ORB Impact")

        # 10 SUMMARY TABLE (IMPROVED)
        axes[9].axis('off')
        axes[9].set_title("Session Summary", color=COLORS['text'])

        runtime = ts[-1]

        summary_data = [
            ["Metric", "Avg", "Min", "Max", "p5", "p95"],

            ["FPS",
             f"{fps.mean():.2f}",
             f"{fps.min():.2f}",
             f"{fps.max():.2f}",
             f"{np.percentile(fps,5):.2f}",
             f"{np.percentile(fps,95):.2f}"],

            ["Confidence",
             f"{s_c.mean():.3f}",
             f"{s_c.min():.3f}",
             f"{s_c.max():.3f}",
             f"{np.percentile(s_c,5):.3f}",
             f"{np.percentile(s_c,95):.3f}"],

            ["Tracks",
             f"{tracks.mean():.2f}",
             f"{tracks.min()}",
             f"{tracks.max()}",
             "-",
             "-"],

            ["ORB Usage",
             f"{orb.mean()*100:.1f}%",
             "-","-","-","-"],

            ["Runtime",
             f"{runtime:.1f}s",
             "-","-","-","-"],

            ["Frames",
             f"{len(ts)}",
             "-","-","-","-"]
        ]

        table = axes[9].table(
            cellText=summary_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.2]*6
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLORS['grid'])

            if row == 0:
                cell.set_facecolor('#334155')
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('white')
            else:
                cell.set_facecolor(COLORS['panel'])
                cell.get_text().set_color(COLORS['text'])

        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()

        print(f"📊 FULL Graph saved → {path}")
        return path


# ==================== STYLE ====================
def _style_axes(axes):
    for ax in axes:
        ax.set_facecolor(COLORS['panel'])
        ax.grid(True, color=COLORS['grid'], alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(colors=COLORS['text'], labelsize=8)

        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])

        ax.title.set_color(COLORS['text'])