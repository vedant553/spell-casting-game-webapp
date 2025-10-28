# Spell-Casting Duel Game

A real-time spell-casting duel game using webcam hand tracking. Draw magical shapes in the air with your finger to cast spells and defeat your opponent!

### Watch the Demo Video: spell game.mp4 in the root directory

## 🎮 Available Versions

This project includes **two versions** of the game:

### 1. Desktop Version (Pygame)
- **File**: `spell_duel.py`
- **Platform**: Standalone desktop application
- **Engine**: Pygame with OpenCV and MediaPipe
- **Best for**: Traditional desktop gaming experience

### 2. Web Version (Browser-based)
- **Folder**: `spell-duel-webapp/`
- **Platform**: Browser-based (any modern web browser)
- **Stack**: FastAPI backend + HTML/CSS/JS frontend with WebSockets
- **Best for**: Easy access, no local installation needed, modern web interface

👉 **See `spell-duel-webapp/README.md` for detailed web version documentation**

---

## Features

- **Real-time hand tracking** using MediaPipe
- **Gesture-based spell casting** with shape recognition
- **Three unique spells**: Lightning Bolt, Shield, and Heal
- **AI opponent** with random attack patterns
- **Visual effects** and animations
- **Mirrored webcam feed** as game background

---

## 🖥️ Desktop Version - Quick Start

### Installation

1. Install Python 3.8 or higher
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### How to Run

```bash
python spell_duel.py
```

## How to Play

### Controls

1. **Point your index finger** (only index extended) to start drawing a spell
2. **Draw one of three shapes**:
   - **Zig-Zag** (multiple up/down movements) → Lightning Bolt (20 damage)
   - **Circle** (closed loop) → Shield (blocks next attack)
   - **V-Shape** (down and up, or up and down) → Heal (15 HP)
3. **Make a fist** to cast the spell
4. Press **R** to restart after game over
5. Press **Q** to quit

### Game Objective

Reduce your opponent's health to 0 before they defeat you!

### Tips

- The AI opponent attacks every 4.5 seconds
- Use Shield strategically before enemy attacks
- Heal when your HP is low
- Draw shapes clearly for better recognition
- Make sure you have good lighting for hand tracking

## Technical Details

- **Resolution**: 1280x720
- **Frame Rate**: 60 FPS
- **Hand Tracking**: MediaPipe Hands solution
- **Shape Recognition**: Custom algorithm with path analysis
- **Game Engine**: Pygame

## Troubleshooting

- **Webcam not detected**: Make sure your webcam is connected and not being used by another application
- **Hand tracking not working**: Ensure good lighting and position your hand clearly in front of the camera
- **Spells not recognized**: Draw shapes more deliberately and make sure to make a fist to trigger recognition
- **Performance issues**: Close other applications and ensure your system meets the requirements

## System Requirements

### Desktop Version
- Python 3.8+
- Webcam
- Windows/Mac/Linux
- 4GB RAM minimum
- Decent CPU for real-time processing

### Web Version
- Python 3.8+ (for server)
- Webcam
- Modern web browser (Chrome, Firefox, Edge, Safari)
- 4GB RAM minimum (8GB recommended)
- Decent CPU for real-time video processing

---

## 🌐 Web Version - Quick Start

For the browser-based version:

```bash
cd spell-duel-webapp
pip install -r requirements.txt
python main.py
```

Then open your browser to `http://localhost:8000`

**Full documentation**: See `spell-duel-webapp/README.md` for complete setup, architecture, and troubleshooting.

---

## Project Structure

```
Game 2/
│
├── spell_duel.py          # Desktop version (Pygame)
├── requirements.txt       # Desktop dependencies
├── README.md             # This file
│
└── spell-duel-webapp/    # Web version
    ├── main.py           # FastAPI backend
    ├── requirements.txt  # Web version dependencies
    ├── README.md         # Web version documentation
    ├── templates/
    │   └── index.html    # Game interface
    └── static/
        ├── style.css     # UI styling
        └── script.js     # WebSocket client
```

---

## Comparison: Desktop vs Web

| Feature | Desktop (Pygame) | Web (FastAPI) |
|---------|-----------------|---------------|
| **Platform** | Desktop app | Browser-based |
| **Installation** | Python + dependencies | Python server + browser |
| **Performance** | 60 FPS | ~30 FPS |
| **UI** | Pygame overlay | Modern web interface |
| **Accessibility** | Local only | Network accessible |
| **Restart** | Press R key | Auto-reset after 5s |
| **Best for** | Desktop gaming | Remote play, demos |

---

## Credits

Created as a demonstration of:
- Computer vision with OpenCV and MediaPipe
- Game development with Python
- Real-time web applications with WebSockets
- Desktop (Pygame) and web (FastAPI) development

**Technologies Used**: Python, OpenCV, MediaPipe, Pygame, FastAPI, WebSockets, HTML5, CSS3, JavaScript

Enjoy casting spells! 🧙‍♂️⚡🛡️💚
