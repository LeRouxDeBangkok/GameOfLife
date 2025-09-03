import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

ROWS = 90
COLS = 160
INTERVAL_MS = 60

grid = np.zeros((ROWS, COLS), dtype=np.uint8)

def neighbors(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) +
        np.roll(a, 1, 1) + np.roll(a, -1, 1) +
        np.roll(np.roll(a, 1, 0), 1, 1) +
        np.roll(np.roll(a, 1, 0), -1, 1) +
        np.roll(np.roll(a, -1, 0), 1, 1) +
        np.roll(np.roll(a, -1, 0), -1, 1)
    )

def step(a: np.ndarray) -> np.ndarray:
    n = neighbors(a)
    survive = (a == 1) & ((n == 2) | (n == 3))
    born = (a == 0) & (n == 3)
    return (survive | born).astype(np.uint8)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
img = ax.imshow(
    grid,
    interpolation='none',
    cmap='gray',
    vmin=0, vmax=1,
    extent = [-0.5, COLS - 0.5, ROWS - 0.5, -0.5],
    origin='upper',
)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.5, COLS - 0.5)
ax.set_ylim(ROWS - 0.5, -0.5)

ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)

ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
x_lines = np.arange(-0.5, COLS - 0.5 + 1, 1)
y_lines = np.arange(-0.5, ROWS - 0.5 + 1, 1)

v_segments = [((x, -0.5), (x, ROWS - 0.5)) for x in x_lines]
h_segments = [((-0.5, y), (COLS - 0.5, y)) for y in y_lines]

grid_v = LineCollection(v_segments, linewidths=0.8, colors='gray', alpha=0.4, zorder=3)
grid_h = LineCollection(h_segments, linewidths=0.8, colors='gray', alpha=0.4, zorder=3)
ax.add_collection(grid_v)
ax.add_collection(grid_h)

running = False
show_grid = True

def on_click(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    col = int(np.clip(np.round(event.xdata), 0, COLS - 1))
    row = int(np.clip(np.round(event.ydata), 0, ROWS - 1))
    grid[row, col] ^= 1
    img.set_data(grid)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)

def on_key(event):
    global running, show_grid
    if event.key == ' ':
        running = not running
    elif event.key == 'g':
        show_grid = not show_grid
        if show_grid:
            grid_v.set_color('gray');  grid_v.set_alpha(0.4)
            grid_h.set_color('gray');  grid_h.set_alpha(0.4)
        else:
            grid_v.set_color('black'); grid_v.set_alpha(0.0)
            grid_h.set_color('black'); grid_h.set_alpha(0.0)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

def animate(_):
    global grid
    if running:
        grid = step(grid)
        img.set_data(grid)
    return (img,)

ani = FuncAnimation(fig, animate, interval=INTERVAL_MS, blit=False)
plt.subplots_adjust(left=0, right =1, top=1, bottom=0)

def on_resize(event):
    fw, fh = fig.canvas.get_width_height()
    target = COLS / ROWS
    win = fw / fh if fh else target
    if win > target:
        h = 1.0
        w = target * (fh / fw)
        ax.set_position([0.5 * (1 - w), 0.0, w, h])
    else:
        w = 1.0
        h = (1 / target) * (fw / fh)
        ax.set_position([0.0, 0.5 * (1 - h), w, h])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('resize_event', on_resize)
on_resize(None)

plt.show()
