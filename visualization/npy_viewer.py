
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pathlib, sys

carpeta = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
files = sorted(carpeta.glob("*.npy"))
if not files:
    print("No hay .npy en", carpeta)
    sys.exit(1)

def prepare(a):
    # (C,H,W) -> (H,W,C)
    if a.ndim == 3 and a.shape[0] in (1,3,4) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1,2,0))
    return a

def normalize(a):
    if a.dtype == np.uint8:
        return a
    a = a.astype(float)
    mn, mx = a.min(), a.max()
    a = (a - mn) / (mx - mn + 1e-12)
    return a

idx = 0
arr = prepare(np.load(files[idx], allow_pickle=False))

fig, ax = plt.subplots(num="NPY Viewer", figsize=(6,6))
im = None

def show_current():
    global im
    a = normalize(arr)
    ax.clear()
    title = f"{files[idx].name} | shape={arr.shape}, dtype={arr.dtype}"
    if a.ndim == 2:
        im = ax.imshow(a, cmap="gray")
    elif a.ndim == 3 and a.shape[2] in (1,3,4):
        if a.shape[2] == 1:
            im = ax.imshow(a.squeeze(2), cmap="gray")
        else:
            im = ax.imshow(a)
    else:
        # Si es stack: muestra primera slice
        first = np.take(a, 0, axis=0)
        im = ax.imshow(first if first.ndim==2 else first.squeeze(), cmap="gray")
        title += " | mostrando slice 0"
    ax.set_title(title)
    ax.axis("off")
    fig.canvas.draw_idle()

def on_key(event):
    global idx, arr
    if event.key in ("right", "down", "n"):
        idx = (idx + 1) % len(files)
    elif event.key in ("left", "up", "p"):
        idx = (idx - 1) % len(files)
    elif event.key in ("0", "home"):
        idx = 0
    elif event.key in ("end",):
        idx = len(files)-1
    else:
        return
    arr = prepare(np.load(files[idx], allow_pickle=False))
    show_current()

fig.canvas.mpl_connect("key_press_event", on_key)
show_current()
plt.show()