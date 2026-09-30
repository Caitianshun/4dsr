"""Static publication-style plots of the completed radial FFT error budget."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from frequency import OUT, ARMS, BANDS, read, write, sha


def main():
    data = read(OUT / 'summary.json'); assert data['status'] == 'completed_frequency_audit'
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    colors = ['#274862', '#419c9c', '#d8a044']
    for camera, ax in zip(['cam00', 'cam01'], axes):
        rows = [next(r for r in data['summaries'] if r['arm'] == a and r['scope'] == camera) for a in ARMS]
        left = np.zeros(len(rows))
        for band, color in zip(BANDS, colors):
            values = np.array([r[band + '_mse'] * 1000 for r in rows])
            ax.barh(np.arange(len(rows)), values, left=left, label=band, color=color, height=.65)
            left += values
        ax.set_yticks(range(len(rows)), ARMS, fontsize=10); ax.invert_yaxis()
        ax.set_xlabel('Absolute band MSE contribution (x 0.001)', fontsize=11)
        ax.set_title(camera + ' | 60 fixed development frames', fontsize=12)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=10)
    path = OUT / 'band_mse_by_camera.png'; fig.savefig(path, dpi=180); plt.close(fig)
    write(OUT / 'figures.json', dict(status='completed', summary_sha256=sha(OUT / 'summary.json'),
           figures=[dict(path=str(path), sha256=sha(path))],
           policy='Absolute MSE contributions; each camera has its own horizontal scale. No error shares portrayed as image content.'))
    print(path)


if __name__ == '__main__': main()
