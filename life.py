import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROWS = 120
COLS = 160
DENSITY = 0.20
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

def step (a: np.ndarray) -> np.ndarray:
    n = neighbors(a)
    survive = (a == 1) & ((n == 2) | (n == 3))
    born = (a == 0) & (n == 3)
    return (survive | born).astype(np.uint8)

fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
img = ax.imshow(
    grid,
    interpolation='none',
    cmap='gray',
    vmin=0, vmax = 1
)
ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xticks([])
ax.set_yticks([])

running = False

def on_click(event):
    if event.inaxes != ax:
        return
    col = int(event.xdata)
    row = int(event.ydata)
    grid[row,col] ^= 1
    img.set_data(grid)
    plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

def on_key(event):
    global running
    if event.key == ' ':
        running = not running

fig.canvas.mpl_connect('key_press_event', on_key)

def animate(_):
    global grid
    if running:
        grid = step(grid)
        img.set_data(grid)
    return (img,)


ani = FuncAnimation(fig, animate, interval=INTERVAL_MS, blit=True)
plt.show()