# Spell-Casting Duel - Web Application

A real-time spell-casting duel game using webcam hand tracking, now playable in your web browser! Draw magical shapes in the air with your finger to cast spells and defeat your opponent.

## 🎥 Gameplay Demo

https://github.com/your-username/spell-duel-webapp/raw/main/spell%20game.mp4

## Features

- **Real-time WebSocket streaming** for instant game updates
- **Browser-based gameplay** - no desktop application required
- **Gesture-based spell casting** with shape recognition using MediaPipe
- **Three unique spells**: Lightning Bolt, Shield, and Heal
- **AI opponent** with random attack patterns
- **Visual effects** drawn directly on the video feed
- **Modern web interface** with responsive sidebar design

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **WebSockets** - Real-time bidirectional communication
- **OpenCV** - Video capture and image processing
- **MediaPipe** - Hand tracking and landmark detection
- **Uvicorn** - ASGI server

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with Flexbox and gradients
- **Vanilla JavaScript** - WebSocket communication and DOM manipulation

## Project Structure

```
/spell-duel-webapp
│
├── main.py              # FastAPI backend with WebSocket endpoint
├── requirements.txt     # Python dependencies
│
├── /templates
│   └── index.html       # Main HTML page with game layout
│
└── /static
    ├── style.css        # CSS styling for layout and UI
    └── script.js        # JavaScript for WebSocket and UI updates
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Setup

1. **Navigate to the project directory:**
   ```bash
   cd spell-duel-webapp
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8000
   ```

3. **Allow webcam access** when prompted by your browser.

4. **Start playing!** The game will automatically connect via WebSocket.

## How to Play

### Controls

1. **Point your index finger** (only index extended) to start drawing a spell
2. **Draw one of three shapes** in the air:
   - **Zig-Zag** (multiple up/down movements) → Lightning Bolt (20 damage)
   - **Circle** (closed loop) → Shield (blocks next attack)
   - **V-Shape** (down and up, or up and down) → Heal (15 HP)
3. **Make a fist** to cast the spell

### Game Objective

Reduce your opponent's health to 0 before they defeat you!

### Tips

- The AI opponent attacks every 4.5 seconds
- Use Shield strategically before enemy attacks
- Heal when your HP is low
- Draw shapes clearly for better recognition
- Ensure good lighting for hand tracking
- The game automatically resets 5 seconds after game over

## Architecture

### Backend Flow
1. FastAPI server captures webcam feed via OpenCV
2. Each frame is processed with MediaPipe for hand landmark detection
3. Hand gestures are recognized (index finger extended or fist)
4. Game logic updates (spell casting, AI actions, HP changes)
5. Visual feedback is drawn on the frame using OpenCV
6. Frame is encoded as JPEG and sent via WebSocket
7. Game state is sent as JSON via the same WebSocket

### Frontend Flow
1. HTML page loads with two-column Flexbox layout
2. JavaScript establishes WebSocket connection to `/ws`
3. On message received:
   - Binary data (video frames) → Update `<img>` element
   - Text data (JSON) → Parse and update sidebar elements
4. UI reflects real-time game state (HP bars, spell messages, etc.)

## API Endpoints

- **GET /** - Serves the main HTML page
- **WebSocket /ws** - Real-time game streaming and state updates
  - Sends: Video frames (binary) and game state (JSON)
  - Receives: Connection management

## Troubleshooting

### Webcam not working
- Ensure your webcam is connected and not being used by another application
- Check browser permissions for camera access
- Try a different browser

### Connection issues
- Make sure the server is running on port 8000
- Check firewall settings
- Try accessing via `127.0.0.1:8000` instead of `localhost:8000`

### Hand tracking not working
- Ensure good lighting conditions
- Position your hand clearly in front of the camera
- Keep your hand within the camera frame
- Make deliberate, clear gestures

### Spells not recognized
- Draw shapes more slowly and deliberately
- Ensure you're making a clear fist to trigger recognition
- Check that only the index finger is extended while drawing

### Performance issues
- Close other browser tabs and applications
- Reduce browser window size
- Check system resources (CPU/RAM usage)

## Development

### Running in development mode with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Modifying the game:
- **Backend logic**: Edit `main.py` (game state, spell effects, AI behavior)
- **UI layout**: Edit `templates/index.html`
- **Styling**: Edit `static/style.css`
- **Client logic**: Edit `static/script.js`

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Decent processor for real-time video processing
- **Browser**: Modern browser with WebSocket support
- **Webcam**: Any standard webcam (720p or higher recommended)

## Future Enhancements

- Multiple players support
- More spell types
- Leaderboard system
- Difficulty levels
- Mobile device support
- Custom game modes

## Credits

Created as a demonstration of computer vision, real-time web communication, and modern web development with Python and JavaScript.

Enjoy casting spells in your browser! ⚡🛡️💚
