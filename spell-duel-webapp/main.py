import asyncio
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import json


app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Player:
    """Represents a player or opponent in the duel."""
    
    def __init__(self, name, max_hp=100):
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.shield_active = False
        
    def take_damage(self, damage):
        """Apply damage to the player, considering shield."""
        if self.shield_active:
            self.shield_active = False
            return 0  # Damage blocked
        else:
            actual_damage = min(damage, self.hp)
            self.hp -= actual_damage
            return actual_damage
    
    def heal(self, amount):
        """Heal the player, not exceeding max HP."""
        old_hp = self.hp
        self.hp = min(self.hp + amount, self.max_hp)
        return self.hp - old_hp
    
    def activate_shield(self):
        """Activate shield for next attack."""
        self.shield_active = True
    
    def is_alive(self):
        """Check if player is still alive."""
        return self.hp > 0


class ShapeRecognizer:
    """Recognizes shapes drawn by the player."""
    
    @staticmethod
    def recognize_shape(points):
        """
        Analyze a path of points and recognize the shape.
        Returns: 'zigzag', 'circle', 'v-shape', or None
        """
        if len(points) < 10:
            return None
        
        # Simplify points by sampling
        simplified = ShapeRecognizer._simplify_path(points, 20)
        
        if len(simplified) < 5:
            return None
        
        # Check for circle first (closed shape)
        if ShapeRecognizer._is_circle(simplified):
            return 'circle'
        
        # Check for zigzag (multiple direction changes)
        if ShapeRecognizer._is_zigzag(simplified):
            return 'zigzag'
        
        # Check for V-shape (single sharp turn)
        if ShapeRecognizer._is_v_shape(simplified):
            return 'v-shape'
        
        return None
    
    @staticmethod
    def _simplify_path(points, target_count):
        """Simplify path by sampling evenly spaced points."""
        if len(points) <= target_count:
            return points
        
        step = len(points) // target_count
        return [points[i] for i in range(0, len(points), step)][:target_count]
    
    @staticmethod
    def _is_circle(points):
        """
        Check if the path forms a circle.
        A circle is detected if:
        1. The end point is close to the start point
        2. The path has reasonable curvature
        """
        if len(points) < 8:
            return False
        
        start = np.array(points[0])
        end = np.array(points[-1])
        
        # Calculate distance between start and end
        distance = np.linalg.norm(end - start)
        
        # Calculate the total path length
        total_length = 0
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
        
        # Circle: end is close to start relative to total path length
        if total_length > 0:
            closure_ratio = distance / total_length
            if closure_ratio < 0.3:  # End is within 30% of path length from start
                return True
        
        return False
    
    @staticmethod
    def _is_zigzag(points):
        """
        Check if the path has multiple sharp vertical direction changes.
        A zigzag has at least 3-4 peaks/valleys.
        """
        if len(points) < 8:
            return False
        
        # Extract y-coordinates
        y_coords = [p[1] for p in points]
        
        # Find peaks and valleys (local extrema)
        direction_changes = 0
        for i in range(1, len(y_coords) - 1):
            # Check if this is a local maximum or minimum
            if (y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1]) or \
               (y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]):
                direction_changes += 1
        
        # Zigzag should have at least 3 direction changes
        return direction_changes >= 3
    
    @staticmethod
    def _is_v_shape(points):
        """
        Check if the path forms a V-shape (one sharp turn).
        """
        if len(points) < 5:
            return False
        
        # Find the point with maximum or minimum y-coordinate (the vertex)
        y_coords = [p[1] for p in points]
        
        # Find potential vertex (min or max y)
        min_idx = y_coords.index(min(y_coords))
        max_idx = y_coords.index(max(y_coords))
        
        # Check if vertex is roughly in the middle (not at edges)
        for vertex_idx in [min_idx, max_idx]:
            if 0.2 * len(points) < vertex_idx < 0.8 * len(points):
                # Calculate angles before and after vertex
                before_segment = np.array(points[vertex_idx]) - np.array(points[0])
                after_segment = np.array(points[-1]) - np.array(points[vertex_idx])
                
                # Check if the segments form a sharp angle
                if np.linalg.norm(before_segment) > 0 and np.linalg.norm(after_segment) > 0:
                    # Normalize vectors
                    before_norm = before_segment / np.linalg.norm(before_segment)
                    after_norm = after_segment / np.linalg.norm(after_segment)
                    
                    # Calculate angle between segments
                    dot_product = np.dot(before_norm, after_norm)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle = math.acos(dot_product)
                    
                    # V-shape should have an angle between 30 and 150 degrees
                    if 0.5 < angle < 2.6:  # radians
                        return True
        
        return False


class GameState:
    """Manages the game state for a WebSocket connection."""
    
    def __init__(self):
        self.player = Player("Player")
        self.opponent = Player("Opponent")
        self.drawing = False
        self.path = []
        self.last_spell = None
        self.spell_display_time = 0
        
        # AI opponent
        self.ai_last_action_time = time.time()
        self.ai_action_interval = 4.5
        self.ai_current_action = None
        self.ai_action_display_time = 0
        
        # Animations
        self.animations = []
        
        # Game state
        self.game_over = False
        self.winner = None
        
    def reset(self):
        """Reset the game state."""
        self.player = Player("Player")
        self.opponent = Player("Opponent")
        self.drawing = False
        self.path = []
        self.last_spell = None
        self.spell_display_time = 0
        self.ai_last_action_time = time.time()
        self.ai_current_action = None
        self.ai_action_display_time = 0
        self.animations = []
        self.game_over = False
        self.winner = None
    
    def update_ai(self):
        """Update AI opponent behavior."""
        current_time = time.time()
        
        if current_time - self.ai_last_action_time >= self.ai_action_interval:
            # AI takes an action
            action = random.choice(['attack', 'attack', 'defend'])
            
            if action == 'attack':
                damage = self.player.take_damage(10)
                self.ai_current_action = f"Opponent attacks! (-{damage} HP)"
                self.animations.append({
                    'type': 'enemy_attack',
                    'start_time': current_time,
                    'duration': 0.5
                })
            else:
                self.ai_current_action = "Opponent defends..."
            
            self.ai_action_display_time = current_time
            self.ai_last_action_time = current_time
            
            # Check win condition
            if not self.player.is_alive():
                self.game_over = True
                self.winner = "Opponent"
    
    def recognize_and_cast_spell(self):
        """Recognize the drawn shape and cast the corresponding spell."""
        shape = ShapeRecognizer.recognize_shape(self.path)
        
        if shape == 'zigzag':
            # Lightning Bolt: 20 damage
            damage = self.opponent.take_damage(20)
            self.last_spell = f"Lightning Bolt! (-{damage} HP)"
            self.animations.append({
                'type': 'lightning',
                'start_time': time.time(),
                'duration': 0.5
            })
            
        elif shape == 'circle':
            # Shield: Block next attack
            self.player.activate_shield()
            self.last_spell = "Shield Activated!"
            self.animations.append({
                'type': 'shield',
                'start_time': time.time(),
                'duration': 0.8
            })
            
        elif shape == 'v-shape':
            # Heal: Restore 15 HP
            healed = self.player.heal(15)
            self.last_spell = f"Heal! (+{healed} HP)"
            self.animations.append({
                'type': 'heal',
                'start_time': time.time(),
                'duration': 0.8
            })
        else:
            self.last_spell = "Spell Failed!"
        
        self.spell_display_time = time.time()
        
        # Check win condition
        if not self.opponent.is_alive():
            self.game_over = True
            self.winner = "Player"
    
    def get_state_json(self):
        """Get current game state as JSON."""
        return {
            "type": "game_state",
            "player_hp": self.player.hp,
            "player_max_hp": self.player.max_hp,
            "player_shield": self.player.shield_active,
            "opponent_hp": self.opponent.hp,
            "opponent_max_hp": self.opponent.max_hp,
            "last_spell": self.last_spell if time.time() - self.spell_display_time < 2.0 else "",
            "ai_action": self.ai_current_action if time.time() - self.ai_action_display_time < 2.5 else "",
            "game_over": self.game_over,
            "winner": self.winner
        }


def is_index_finger_extended(hand_landmarks):
    """Check if only the index finger is extended."""
    landmarks = hand_landmarks.landmark
    
    # Index finger tip vs MCP (base)
    index_extended = landmarks[8].y < landmarks[6].y
    
    # Other fingers should be curled
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    
    return index_extended and middle_curled and ring_curled and pinky_curled


def is_fist(hand_landmarks):
    """Check if the hand is making a fist."""
    landmarks = hand_landmarks.landmark
    
    # All fingers should be curled
    index_curled = landmarks[8].y > landmarks[6].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    
    return index_curled and middle_curled and ring_curled and pinky_curled


def draw_trail_on_frame(frame, path):
    """Draw the spell-casting trail on the video frame."""
    if len(path) > 1:
        # Draw glowing trail
        for i in range(1, len(path)):
            thickness = max(1, 8 - i // 10)
            cv2.line(frame, path[i-1], path[i], (255, 0, 200), thickness)
        
        # Draw dots at each point
        for point in path[::3]:  # Every 3rd point
            cv2.circle(frame, point, 4, (255, 255, 0), -1)


def draw_animations_on_frame(frame, animations, width, height):
    """Draw spell animations on the video frame."""
    current_time = time.time()
    active_animations = []
    
    for anim in animations:
        elapsed = current_time - anim['start_time']
        if elapsed < anim['duration']:
            active_animations.append(anim)
            
            if anim['type'] == 'lightning':
                # Lightning bolt from player to opponent
                start_pos = (width // 4, height - 150)
                end_pos = (3 * width // 4, 150)
                
                # Draw jagged lightning
                points = [start_pos]
                num_segments = 8
                for i in range(1, num_segments):
                    t = i / num_segments
                    x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t + random.randint(-30, 30))
                    y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t + random.randint(-30, 30))
                    points.append((x, y))
                points.append(end_pos)
                
                # Draw lightning with yellow/white
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i+1], (0, 255, 255), 5)
                    cv2.line(frame, points[i], points[i+1], (255, 255, 255), 2)
                
            elif anim['type'] == 'shield':
                # Blue shield circle around player
                cv2.circle(frame, (width // 4, height - 200), 120, (255, 100, 0), 8)
                
            elif anim['type'] == 'heal':
                # Green sparkles around player
                for _ in range(5):
                    x = width // 4 + random.randint(-80, 80)
                    y = height - 200 + random.randint(-80, 80)
                    size = random.randint(3, 8)
                    cv2.circle(frame, (x, y), size, (0, 255, 0), -1)
                
            elif anim['type'] == 'enemy_attack':
                # Red slash from opponent to player
                start_pos = (3 * width // 4, 150)
                end_pos = (width // 4, height - 150)
                cv2.line(frame, start_pos, end_pos, (0, 0, 255), 8)
                cv2.line(frame, start_pos, end_pos, (0, 165, 255), 4)
    
    return active_animations


def draw_text_on_frame(frame, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2):
    """Draw text with shadow on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw shadow
    cv2.putText(frame, text, (position[0] + 2, position[1] + 2), font, font_scale, (0, 0, 0), thickness + 1)
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time game streaming."""
    await websocket.accept()
    
    # Initialize game state
    game_state = GameState()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and not game_state.game_over:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)
                
                # Check gestures
                if is_index_finger_extended(hand_landmarks):
                    # Start or continue drawing
                    if not game_state.drawing:
                        game_state.drawing = True
                        game_state.path = []
                    
                    game_state.path.append((x, y))
                    
                elif is_fist(hand_landmarks) and game_state.drawing:
                    # Stop drawing and recognize shape
                    game_state.drawing = False
                    if len(game_state.path) > 10:
                        game_state.recognize_and_cast_spell()
                    game_state.path = []
            else:
                # No hand detected, reset drawing
                if game_state.drawing and len(game_state.path) > 10:
                    game_state.recognize_and_cast_spell()
                game_state.drawing = False
                game_state.path = []
            
            # Update AI
            if not game_state.game_over:
                game_state.update_ai()
            
            # Draw trail on frame
            if game_state.drawing:
                draw_trail_on_frame(frame, game_state.path)
            
            # Draw animations
            game_state.animations = draw_animations_on_frame(frame, game_state.animations, width, height)
            
            # Draw spell feedback
            if game_state.last_spell and time.time() - game_state.spell_display_time < 2.0:
                draw_text_on_frame(frame, game_state.last_spell, (width // 2 - 200, height // 2 - 100), 1.2, (0, 255, 255), 3)
            
            # Draw AI action
            if game_state.ai_current_action and time.time() - game_state.ai_action_display_time < 2.5:
                draw_text_on_frame(frame, game_state.ai_current_action, (width // 2 - 200, 150), 0.8, (0, 165, 255), 2)
            
            # Draw game over text
            if game_state.game_over:
                if game_state.winner == "Player":
                    draw_text_on_frame(frame, "YOU WIN!", (width // 2 - 150, height // 2), 2.0, (0, 255, 0), 4)
                else:
                    draw_text_on_frame(frame, "GAME OVER", (width // 2 - 200, height // 2), 2.0, (0, 0, 255), 4)
                draw_text_on_frame(frame, "Game will reset in 5 seconds...", (width // 2 - 300, height // 2 + 80), 0.8, (255, 255, 255), 2)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_bytes = buffer.tobytes()
            
            # Send video frame
            await websocket.send_bytes(jpg_bytes)
            
            # Send game state as JSON
            state_json = game_state.get_state_json()
            await websocket.send_text(json.dumps(state_json))
            
            # Auto-reset after game over
            if game_state.game_over:
                await asyncio.sleep(5)
                game_state.reset()
            
            # Small delay to control frame rate
            await asyncio.sleep(1/30)  # ~30 FPS
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        hands.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
