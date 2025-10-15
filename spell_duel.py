import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import random
import time
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1280, 720
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
PURPLE = (200, 0, 255)
ORANGE = (255, 165, 0)


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


class Game:
    """Main game class managing the spell-casting duel."""
    
    def __init__(self):
        # Pygame setup
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Spell-Casting Duel")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # OpenCV and MediaPipe setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Game state
        self.player = Player("Player")
        self.opponent = Player("Opponent")
        self.running = True
        self.game_over = False
        self.winner = None
        
        # Drawing state
        self.drawing = False
        self.path = []
        self.last_spell = None
        self.spell_display_time = 0
        
        # AI opponent
        self.ai_last_action_time = time.time()
        self.ai_action_interval = 4.5  # seconds
        self.ai_current_action = None
        self.ai_action_display_time = 0
        
        # Animation state
        self.animations = []
        
    def _is_index_finger_extended(self, hand_landmarks):
        """
        Check if only the index finger is extended.
        This is used as the trigger for drawing mode.
        """
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Index finger tip vs MCP (base)
        index_extended = landmarks[8].y < landmarks[6].y
        
        # Other fingers should be curled
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y
        
        return index_extended and middle_curled and ring_curled and pinky_curled
    
    def _is_fist(self, hand_landmarks):
        """
        Check if the hand is making a fist.
        This is used as the trigger to stop drawing and recognize the shape.
        """
        landmarks = hand_landmarks.landmark
        
        # All fingers should be curled
        index_curled = landmarks[8].y > landmarks[6].y
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y
        
        return index_curled and middle_curled and ring_curled and pinky_curled
    
    def process_hand_tracking(self, frame):
        """
        Process hand tracking and update drawing state.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get index finger tip position (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)
            
            # Check gestures
            if self._is_index_finger_extended(hand_landmarks):
                # Start or continue drawing
                if not self.drawing:
                    self.drawing = True
                    self.path = []
                
                self.path.append((x, y))
                
            elif self._is_fist(hand_landmarks) and self.drawing:
                # Stop drawing and recognize shape
                self.drawing = False
                if len(self.path) > 10:
                    self.recognize_and_cast_spell()
                self.path = []
        else:
            # No hand detected, reset drawing
            if self.drawing and len(self.path) > 10:
                self.recognize_and_cast_spell()
            self.drawing = False
            self.path = []
    
    def recognize_and_cast_spell(self):
        """
        Recognize the drawn shape and cast the corresponding spell.
        """
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
    
    def update_ai(self):
        """
        Update AI opponent behavior.
        """
        current_time = time.time()
        
        if current_time - self.ai_last_action_time >= self.ai_action_interval:
            # AI takes an action
            action = random.choice(['attack', 'attack', 'defend'])  # 2/3 attack, 1/3 defend
            
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
    
    def draw_health_bars(self):
        """
        Draw health bars for player and opponent.
        """
        # Player health bar (bottom left)
        player_bar_x, player_bar_y = 50, HEIGHT - 80
        bar_width, bar_height = 300, 30
        
        # Background
        pygame.draw.rect(self.screen, RED, (player_bar_x, player_bar_y, bar_width, bar_height))
        # Health
        player_health_width = int((self.player.hp / self.player.max_hp) * bar_width)
        pygame.draw.rect(self.screen, GREEN, (player_bar_x, player_bar_y, player_health_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, WHITE, (player_bar_x, player_bar_y, bar_width, bar_height), 3)
        
        # Player label and HP text
        label = self.font_small.render("Player", True, WHITE)
        self.screen.blit(label, (player_bar_x, player_bar_y - 35))
        hp_text = self.font_small.render(f"{self.player.hp}/{self.player.max_hp}", True, WHITE)
        self.screen.blit(hp_text, (player_bar_x + bar_width + 10, player_bar_y - 5))
        
        # Shield indicator
        if self.player.shield_active:
            shield_text = self.font_small.render("[SHIELD]", True, CYAN)
            self.screen.blit(shield_text, (player_bar_x, player_bar_y + bar_height + 5))
        
        # Opponent health bar (top right)
        opponent_bar_x = WIDTH - bar_width - 50
        opponent_bar_y = 50
        
        # Background
        pygame.draw.rect(self.screen, RED, (opponent_bar_x, opponent_bar_y, bar_width, bar_height))
        # Health
        opponent_health_width = int((self.opponent.hp / self.opponent.max_hp) * bar_width)
        pygame.draw.rect(self.screen, GREEN, (opponent_bar_x, opponent_bar_y, opponent_health_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, WHITE, (opponent_bar_x, opponent_bar_y, bar_width, bar_height), 3)
        
        # Opponent label and HP text
        label = self.font_small.render("Opponent", True, WHITE)
        self.screen.blit(label, (opponent_bar_x, opponent_bar_y - 35))
        hp_text = self.font_small.render(f"{self.opponent.hp}/{self.opponent.max_hp}", True, WHITE)
        self.screen.blit(hp_text, (opponent_bar_x + bar_width + 10, opponent_bar_y - 5))
    
    def draw_trail(self):
        """
        Draw the spell-casting trail while the player is drawing.
        """
        if len(self.path) > 1:
            # Draw glowing trail
            for i in range(1, len(self.path)):
                thickness = max(1, 8 - i // 10)
                pygame.draw.line(self.screen, PURPLE, self.path[i-1], self.path[i], thickness)
            
            # Draw dots at each point
            for point in self.path[::3]:  # Every 3rd point
                pygame.draw.circle(self.screen, CYAN, point, 4)
    
    def draw_spell_feedback(self):
        """
        Display spell name when cast.
        """
        if self.last_spell and time.time() - self.spell_display_time < 2.0:
            text = self.font_medium.render(self.last_spell, True, YELLOW)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
            
            # Draw shadow
            shadow = self.font_medium.render(self.last_spell, True, BLACK)
            shadow_rect = shadow.get_rect(center=(WIDTH // 2 + 2, HEIGHT // 2 - 98))
            self.screen.blit(shadow, shadow_rect)
            
            self.screen.blit(text, text_rect)
    
    def draw_ai_action(self):
        """
        Display AI action text.
        """
        if self.ai_current_action and time.time() - self.ai_action_display_time < 2.5:
            text = self.font_small.render(self.ai_current_action, True, ORANGE)
            text_rect = text.get_rect(center=(WIDTH // 2, 150))
            self.screen.blit(text, text_rect)
    
    def draw_animations(self):
        """
        Draw spell animations.
        """
        current_time = time.time()
        active_animations = []
        
        for anim in self.animations:
            elapsed = current_time - anim['start_time']
            if elapsed < anim['duration']:
                active_animations.append(anim)
                
                if anim['type'] == 'lightning':
                    # Lightning bolt from player to opponent
                    start_pos = (WIDTH // 4, HEIGHT - 150)
                    end_pos = (3 * WIDTH // 4, 150)
                    
                    # Draw jagged lightning
                    points = [start_pos]
                    num_segments = 8
                    for i in range(1, num_segments):
                        t = i / num_segments
                        x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t + random.randint(-30, 30))
                        y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t + random.randint(-30, 30))
                        points.append((x, y))
                    points.append(end_pos)
                    
                    pygame.draw.lines(self.screen, YELLOW, False, points, 5)
                    pygame.draw.lines(self.screen, WHITE, False, points, 2)
                    
                elif anim['type'] == 'shield':
                    # Blue shield circle around player
                    alpha = int(150 * (1 - elapsed / anim['duration']))
                    shield_surface = pygame.Surface((300, 300), pygame.SRCALPHA)
                    pygame.draw.circle(shield_surface, (*BLUE, alpha), (150, 150), 120, 8)
                    self.screen.blit(shield_surface, (WIDTH // 4 - 150, HEIGHT - 300))
                    
                elif anim['type'] == 'heal':
                    # Green sparkles around player
                    for _ in range(5):
                        x = WIDTH // 4 + random.randint(-80, 80)
                        y = HEIGHT - 200 + random.randint(-80, 80)
                        size = random.randint(3, 8)
                        pygame.draw.circle(self.screen, GREEN, (x, y), size)
                    
                elif anim['type'] == 'enemy_attack':
                    # Red slash from opponent to player
                    start_pos = (3 * WIDTH // 4, 150)
                    end_pos = (WIDTH // 4, HEIGHT - 150)
                    pygame.draw.line(self.screen, RED, start_pos, end_pos, 8)
                    pygame.draw.line(self.screen, ORANGE, start_pos, end_pos, 4)
        
        self.animations = active_animations
    
    def draw_instructions(self):
        """
        Draw game instructions.
        """
        instructions = [
            "Draw spells with your index finger:",
            "Zig-Zag = Lightning (20 dmg)",
            "Circle = Shield (block 1 attack)",
            "V-Shape = Heal (15 HP)",
            "Make a fist to cast!"
        ]
        
        y_offset = HEIGHT - 200
        for i, instruction in enumerate(instructions):
            if i == 0:
                text = self.font_small.render(instruction, True, WHITE)
            else:
                text = self.font_small.render(instruction, True, CYAN)
            self.screen.blit(text, (50, y_offset + i * 30))
    
    def draw_game_over(self):
        """
        Draw game over screen.
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        if self.winner == "Player":
            text = self.font_large.render("YOU WIN!", True, GREEN)
        else:
            text = self.font_large.render("GAME OVER", True, RED)
        
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        self.screen.blit(text, text_rect)
        
        # Restart instruction
        restart_text = self.font_medium.render("Press R to Restart or Q to Quit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
    
    def reset_game(self):
        """
        Reset the game state.
        """
        self.player = Player("Player")
        self.opponent = Player("Opponent")
        self.game_over = False
        self.winner = None
        self.path = []
        self.drawing = False
        self.last_spell = None
        self.ai_last_action_time = time.time()
        self.ai_current_action = None
        self.animations = []
    
    def run(self):
        """
        Main game loop.
        """
        while self.running:
            self.clock.tick(FPS)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_q:
                            self.running = False
            
            # Capture webcam frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert frame to Pygame surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Draw background (webcam feed)
            self.screen.blit(frame_surface, (0, 0))
            
            if not self.game_over:
                # Process hand tracking
                self.process_hand_tracking(frame)
                
                # Update AI
                self.update_ai()
                
                # Check win condition
                if not self.player.is_alive():
                    self.game_over = True
                    self.winner = "Opponent"
                elif not self.opponent.is_alive():
                    self.game_over = True
                    self.winner = "Player"
                
                # Draw game elements
                self.draw_trail()
                self.draw_animations()
                self.draw_health_bars()
                self.draw_spell_feedback()
                self.draw_ai_action()
                self.draw_instructions()
            else:
                self.draw_game_over()
            
            # Update display
            pygame.display.flip()
        
        # Cleanup
        self.cap.release()
        self.hands.close()
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
