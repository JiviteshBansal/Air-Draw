# Air-Draw
# ✋ Air Draw — Gesture-Controlled Virtual Whiteboard

Ever wanted to draw without touching anything? Air Draw lets you paint, erase,
and switch colors in real time using nothing but your hand in front of a webcam.

Built with PyTorch and MediaPipe, it tracks 21 hand keypoints at 25–35 FPS on CPU
and turns finger gestures into drawing commands — no special hardware needed.

## ✨ How it works

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw |
| 🖐️ All 5 fingers | Erase |
| 👍 Thumb only | Switch to Blue |
| 🖕 Middle finger only | Switch to Yellow |
| 💍 Ring finger only | Switch to Red |
| 🤙 Pinky only | Switch to Green |

## 🛠️ Tech Stack
- **PyTorch** — landmark post-processing as a proper `nn.Module`
- **MediaPipe** — 21-keypoint hand landmark detection
- **OpenCV** — webcam input and real-time rendering
- **NumPy** — off-screen canvas management

## ⌨️ Controls
`ESC` quit &nbsp;|&nbsp; `C` clear &nbsp;|&nbsp; `S` save PNG &nbsp;|&nbsp; `↑↓` brush size &nbsp;|&nbsp; `←→` eraser size &nbsp;|&nbsp; `L` toggle skeleton
