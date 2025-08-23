import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROWS = 120
COLS = 160
DENSITY = 0.20
INTERVAL_MS = 60

rng = np.random.default_rng()
grid = rng.random((ROWS, COLS)) < DENSITY
grid = grid.astype(np.uint8)

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
ax.axis('off')

def animate(_):
    global grid
    grid = step(grid)
    img.set_data(grid)
    return (img,)

ani = FuncAnimation(fig, animate, interval=INTERVAL_MS, blit=True)
plt.show()