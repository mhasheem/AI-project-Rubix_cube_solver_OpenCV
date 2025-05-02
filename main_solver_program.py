
import cv2
import numpy as np
import json
import os
import time
import kociemba 
from collections import deque

class RubiksCubeDetector:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.cube_state = {
            'up': [''] * 9,
            'right': [''] * 9,
            'front': [''] * 9,
            'down': [''] * 9,
            'left': [''] * 9,
            'back': [''] * 9
        }

        self.current_face = 'front'
        self.faces_captured = set()
        self.running = True
        self.grid_size = 3  
        self.show_help = True
        
        self.color_names = ['yellow', 'red', 'orange', 'green', 'blue', 'white']

        self.solution = []
        self.current_move_index = -1
        self.solving_mode = False
        self.expected_state_after_moves = []
        self.current_expected_state = {}

        self.load_color_samples()
        self.center_colors = {
            'up': 'white',     
            'right': 'orange',   
            'front': 'green',  
            'down': 'yellow',  
            'left': 'red',  
            'back': 'blue'     
        }

        self.color_to_letter = {
            'white': 'U',   
            'red': 'L',     
            'green': 'F',   
            'yellow': 'D',  
            'orange': 'R',  
            'blue': 'B'     
        }
        self.letter_to_color = {v: k for k, v in self.color_to_letter.items()}
        self.face_display_colors = {
            'up': (255, 255, 255),    
            'right': (0, 0, 255),     
            'front': (0, 255, 0),     
            'down': (0, 255, 255),    
            'left': (0, 165, 255),    
            'back': (255, 0, 0)       
        }

        self.MOVE_DESCRIPTIONS = {
            'U': "Rotate the top face clockwise",
            'U\'': "Rotate the top face counter-clockwise",
            'U2': "Rotate the top face 180 degrees",
            'D': "Rotate the bottom face clockwise",
            'D\'': "Rotate the bottom face counter-clockwise",
            'D2': "Rotate the bottom face 180 degrees",
            'R': "Rotate the right face clockwise",
            'R\'': "Rotate the right face counter-clockwise",
            'R2': "Rotate the right face 180 degrees",
            'L': "Rotate the left face clockwise",
            'L\'': "Rotate the left face counter-clockwise",
            'L2': "Rotate the left face 180 degrees",
            'F': "Rotate the front face clockwise",
            'F\'': "Rotate the front face counter-clockwise",
            'F2': "Rotate the front face 180 degrees",
            'B': "Rotate the back face clockwise",
            'B\'': "Rotate the back face counter-clockwise",
            'B2': "Rotate the back face 180 degrees"
        }
        

    def load_color_samples(self):

        self.color_samples = {}
        self.color_ranges = {}
        try:
            with open("cube_colors.txt", "r") as f:
                lines = f.readlines()
            data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
            for line in data_lines:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                _, color_name, _, _, _, _, _, h, s, v = parts
                hsv = np.array([int(h), int(s), int(v)])
                self.color_samples.setdefault(color_name, []).append(hsv)
            if not self.color_samples:
                raise ValueError("No samples found in cube_colors.txt")
            
            for cname, hsv_list in self.color_samples.items():
                arr = np.stack(hsv_list, axis=0)  
                lo = arr.min(axis=0) - np.array([10, 50, 50])
                hi = arr.max(axis=0) + np.array([10, 50, 50])
                lo = np.clip(lo, [0, 0, 0], [180, 255, 255])
                hi = np.clip(hi, [0, 0, 0], [180, 255, 255])
                self.color_ranges[cname] = (lo.astype(int), hi.astype(int))
                print(f"{cname}: HSV range {lo.astype(int)} → {hi.astype(int)}")
        except Exception as e:
            print(f"Error loading color samples: {e}")
            print("Using default color ranges instead")
            self.color_ranges = {
                'white':  (np.array([0,   0,   150]), np.array([180, 30,  255])),
                'yellow': (np.array([20, 100, 100]), np.array([30,  255, 255])),
                'red':    (np.array([0,   100, 100]), np.array([10,  255, 255])),
                'orange': (np.array([10,  100, 100]), np.array([20,  255, 255])),
                'blue':   (np.array([100, 100, 100]), np.array([130, 255, 255])),
                'green':  (np.array([40,  100, 100]), np.array([80,  255, 255])),
            }

    
    def create_color_ranges(self):
        if len(self.color_samples) < 6:
            print("Warning: Less than 6 color samples found. Some colors may be missing.")

        h_tolerance = 10  
        s_tolerance = 50
        v_tolerance = 50

        for i, (_, hsv) in enumerate(self.color_samples):
            if i >= len(self.color_names):
                break
            color_name = self.color_names[i]
            lower = (max(0, hsv[0] - h_tolerance), 
                    max(0, hsv[1] - s_tolerance), 
                    max(0, hsv[2] - v_tolerance))
            
            upper = (min(180, hsv[0] + h_tolerance), 
                    min(255, hsv[1] + s_tolerance), 
                    min(255, hsv[2] + v_tolerance))
            
            self.color_ranges[color_name] = (np.array(lower), np.array(upper))
            print(f"Color {color_name}: HSV={hsv}, Range={lower} to {upper}")

    def start_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        return True
    
    def detect_grid(self, frame):
        
        height, width = frame.shape[:2]
        margin_x = width // 4
        margin_y = height // 4
        grid_width = width - 2 * margin_x
        grid_height = height - 2 * margin_y
        cell_width = grid_width // self.grid_size
        cell_height = grid_height // self.grid_size
        grid_points = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = margin_x + col * cell_width
                y = margin_y + row * cell_height
                center_x = x + cell_width // 2
                center_y = y + cell_height // 2
                grid_points.append((center_x, center_y))
        return grid_points, cell_width, cell_height
    
    def detect_colors(self, frame, grid_points):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colors = []
        for point in grid_points:
            x, y = point
            roi_size = 5
            roi = hsv_frame[y-roi_size:y+roi_size, x-roi_size:x+roi_size]
            if roi.size == 0: 
                colors.append(None)
                continue
            avg_hsv = np.mean(roi, axis=(0, 1))
            detected_color = self.classify_color(avg_hsv)
            colors.append(detected_color)
        return colors
    
    def classify_color(self, hsv_value):
        best_match = None
        min_distance = float('inf')
        for color_name, (lower, upper) in self.color_ranges.items():
            if np.all(hsv_value >= lower) and np.all(hsv_value <= upper):
                center = (lower + upper) / 2
                distance = np.sum((hsv_value - center) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_match = color_name
        
        if best_match is None:
            for color_name, (lower, upper) in self.color_ranges.items():
                center = (lower + upper) / 2
                distance = np.sum((hsv_value - center) ** 2)        
                if distance < min_distance:
                    min_distance = distance
                    best_match = color_name
        return best_match
    
    def draw_grid(self, frame, grid_points, cell_width, cell_height, colors):
        for i, (point, color) in enumerate(zip(grid_points, colors)):
            x, y = point
            half_w = cell_width // 2
            half_h = cell_height // 2
            cv2.rectangle(frame, (x - half_w, y - half_h), (x + half_w, y + half_h), (0, 0, 0), 2)
            
            if color:
                color_bgr = self.color_to_bgr(color)
                cv2.rectangle(frame, 
                              (x - half_w + 2, y - half_h + 2), (x + half_w - 2, y + half_h - 2), color_bgr, -1)
                cv2.putText(frame, color[0].upper(), (x - 10, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def color_to_bgr(self, color_name):
        color_map = {
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0)
        }
        return color_map.get(color_name, (128, 128, 128))
    
    def draw_face_indicators(self, frame):
        faces = ['up', 'right', 'front', 'down', 'left', 'back']
        spacing = 20  
        font_scale = 0.5  
        for i, face in enumerate(faces):
            color = (0, 255, 0) if face == self.current_face else (255, 255, 255)
            thickness = 2 if face == self.current_face else 1
            if face in self.faces_captured:
                cv2.rectangle(frame, (20, 40 + i*spacing - 10), (80, 40 + i*spacing + 10), (0, 0, 255), -1)
            cv2.putText(frame, face.upper(), (30, 40 + i*spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
    def draw_help(self, frame):
        if self.solving_mode:
            help_text = [
                "- SPACE: Next move",
                "- P: Previous move"
            ]
            y_offset = 70  
            x_position = 20  
        else:
            help_text = [
                "Controls:",
                "- SPACE: Capture current face",
                "- U/R/F/D/L/B: Select face (Up/Right/Front/Down/Left/Back)",
                "- C: Clear all captured faces",
                "- S: Save cube state",
                "- Z: Solve cube (requires all faces captured)",
                "- Q: Quit",
                "- ?: Toggle help text"
            ]
            y_offset = frame.shape[0] - 150
            x_position = frame.shape[1] - 300
        font_scale = 0.5
        line_height = 20
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (x_position, y_offset + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    def save_cube_state(self):
        if len(self.faces_captured) < 6:
            print(f"Warning: Only {len(self.faces_captured)} faces captured out of 6")
            missing_faces = set(['up', 'right', 'front', 'down', 'left', 'back']) - self.faces_captured
            print(f"Missing faces: {', '.join(missing_faces)}")
            print("Do you want to save the incomplete cube state? (y/n)")
            confirm = input().lower()
            if confirm != 'y':
                print("Save cancelled")
                return

        cube_state_letters = {}
        for face, colors in self.cube_state.items():
            cube_state_letters[face] = ''.join(self.color_to_letter.get(color, 'X') for color in colors)
        with open('cube_state.json', 'w') as f:
            json.dump(cube_state_letters, f, indent=4)
        print("Cube state saved to cube_state.json")
        with open('cube_state_readable.txt', 'w') as f:
            f.write("Rubik's Cube State\n")
            f.write("=================\n\n")    
            for face, colors in self.cube_state.items():
                f.write(f"{face.upper()} face:\n")
                for i in range(0, 9, 3):
                    f.write(f"  {colors[i]:8} {colors[i+1]:8} {colors[i+2]:8}\n")
                f.write("\n")
        print("Readable cube state saved to cube_state_readable.txt")
    
    def capture_current_face(self, colors):
        self.cube_state[self.current_face] = colors
        self.faces_captured.add(self.current_face)
        print(f"Captured {self.current_face} face: {colors}")
    
    def load_cube_state(self, state_file=None):
        self.cube_state = {}
        self.faces_captured.clear()
        if state_file and os.path.exists(state_file):
            with open(state_file, 'r') as f:
                cube_state_letters = json.load(f)
            for face, letters in cube_state_letters.items():
                self.cube_state[face] = [
                    self.letter_to_color.get(letter, '') for letter in letters
                ]
                self.faces_captured.add(face)
            print(f"Loaded cube state from {state_file}")

        else:
            for face in ['up', 'right', 'front', 'down', 'left', 'back']:
                self.cube_state[face] = [None] * 9
            print("Starting with blank cube state (all faces will be captured manually)")

    def verify_cube_state(self):
        if len(self.faces_captured) < 6:
            print(f"Incomplete cube: only {len(self.faces_captured)} faces captured")
            return False
        color_count = {}
        for face, colors in self.cube_state.items():
            for color in colors:
                color_count[color] = color_count.get(color, 0) + 1
        for color, expected_color in self.center_colors.items():
            count = color_count.get(expected_color, 0)
            if count != 9:
                print(f"Color '{expected_color}' appears {count} times, expected 9")
                return False
        for color in color_count:
            if color not in self.color_to_letter:
                print(f"Unknown color detected: {color}")
                return False
        return True
    
    def get_kociemba_string(self):
        if not self.verify_cube_state():
            print("Cannot generate Kociemba string: invalid cube state")
            return None
        kociemba_faces = ['up', 'right', 'front', 'down', 'left', 'back']
        kociemba_string = ''
        for face in kociemba_faces:
            face_colors = self.cube_state[face]
            face_letters = ''.join(self.color_to_letter.get(color, 'X') for color in face_colors)
            kociemba_string += face_letters
        if len(kociemba_string) != 54:
            print(f"Invalid Kociemba string length: {len(kociemba_string)}, expected 54")
            return None
        if 'X' in kociemba_string:
            print("Unknown colors detected in cube state")
            return None
        letter_count = {}
        for letter in kociemba_string:
            letter_count[letter] = letter_count.get(letter, 0) + 1
        for letter, count in letter_count.items():
            if count != 9:
                print(f"Letter {letter} appears {count} times, expected 9")
                return None
        print(f"Generated valid Kociemba string: {kociemba_string}")
        return kociemba_string
    
    def solve_cube(self):
        kociemba_string = self.get_kociemba_string()
        if not kociemba_string:
            print("Cannot solve: invalid cube state")
            return False
        try:
            print(f"Solving cube with state: {kociemba_string}")
            solution = kociemba.solve(kociemba_string)
            self.solution = solution.split()
            print(f"Solution found: {solution}")
            self.generate_expected_states()
            return True
        except Exception as e:
            print(f"Error solving cube: {e}")
            print("Cube state by face:")
            for face, colors in self.cube_state.items():
                print(f"{face}: {colors}")
            return False
    
    def generate_expected_states(self):
        current_state = {face: colors[:] for face, colors in self.cube_state.items()}
        self.expected_state_after_moves = [current_state.copy()]
        for move in self.solution:
            next_state = self.apply_move_to_state(current_state, move)
            self.expected_state_after_moves.append(next_state)
            current_state = next_state.copy()

    def apply_move_to_state(self, state, move):
        new_state = {face: colors[:] for face, colors in state.items()}
        face_map = {'U': 'up', 'R': 'right', 'F': 'front', 
                   'D': 'down', 'L': 'left', 'B': 'back'}
        base_move = move[0]
        face = face_map[base_move]
        if len(move) == 1:
            direction = 'clockwise'
        elif move[1] == "'":
            direction = 'counterclockwise'
        elif move[1] == "2":
            direction = 'half_turn'
        new_state = self.rotate_face(new_state, face, direction)
        return new_state
    
    def rotate_face(self, state, face, direction):
        new_state = {f: colors[:] for f, colors in state.items()}
        face_indices = [0, 1, 2, 5, 8, 7, 6, 3]  
        center_index = 4  
        face_colors = new_state[face][:]
        if direction == 'clockwise':
            shift = 2  
        elif direction == 'counterclockwise':
            shift = -2  
        elif direction == 'half_turn':
            shift = 4  
        temp_colors = [face_colors[i] for i in face_indices]
        for i, idx in enumerate(face_indices):
            new_state[face][idx] = temp_colors[(i - shift) % len(face_indices)]
        if face == 'up':
            self.rotate_adjacent_up(new_state, direction)
        elif face == 'right':
            self.rotate_adjacent_right(new_state, direction)
        elif face == 'front':
            self.rotate_adjacent_front(new_state, direction)
        elif face == 'down':
            self.rotate_adjacent_down(new_state, direction)
        elif face == 'left':
            self.rotate_adjacent_left(new_state, direction)
        elif face == 'back':
            self.rotate_adjacent_back(new_state, direction)
        return new_state
    
    def rotate_adjacent_up(self, state, direction):
        front_top = state['front'][:3]
        right_top = state['right'][:3]
        back_top  = state['back'][:3]
        left_top  = state['left'][:3]
        if direction == 'clockwise':
            state['front'][:3] = right_top
            state['right'][:3] = back_top
            state['back'][:3]  = left_top
            state['left'][:3]  = front_top
        elif direction == 'counterclockwise':
            state['front'][:3] = left_top
            state['left'][:3]  = back_top
            state['back'][:3]  = right_top
            state['right'][:3] = front_top
        elif direction == 'half_turn':
            state['front'][:3], state['back'][:3] = back_top, front_top
            state['right'][:3], state['left'][:3] = left_top, right_top
        return state

    def rotate_adjacent_right(self, state, direction):
        up_right    = [state['up'][2],    state['up'][5],    state['up'][8]]
        front_right = [state['front'][2], state['front'][5], state['front'][8]]
        down_right  = [state['down'][2],  state['down'][5],  state['down'][8]]
        back_left   = [state['back'][6],  state['back'][3],  state['back'][0]]  

        if direction == 'clockwise':
            state['up'][2], state['up'][5], state['up'][8]       = back_left
            state['front'][2], state['front'][5], state['front'][8] = up_right
            state['down'][2], state['down'][5], state['down'][8]     = front_right
            state['back'][6], state['back'][3], state['back'][0]     = down_right
        elif direction == 'counterclockwise':
            state['up'][2], state['up'][5], state['up'][8]       = front_right
            state['front'][2], state['front'][5], state['front'][8] = down_right
            state['down'][2], state['down'][5], state['down'][8]     = back_left
            state['back'][6], state['back'][3], state['back'][0]     = up_right
        elif direction == 'half_turn':
            state['up'][2], state['up'][5], state['up'][8]= down_right
            state['down'][2], state['down'][5], state['down'][8]= up_right
            state['front'][2], state['front'][5], state['front'][8]= back_left
            state['back'][6], state['back'][3], state['back'][0]= front_right
        return state

    def rotate_adjacent_front(self, state, direction):
        up_bottom= [state['up'][6],state['up'][7],state['up'][8]]
        right_left= [state['right'][0],state['right'][3],state['right'][6]]
        down_top= [state['down'][2],state['down'][1],state['down'][0]]  
        left_right= [state['left'][8],state['left'][5],state['left'][2]]  
        if direction == 'clockwise':
            state['up'][6],state['up'][7],state['up'][8]= left_right
            state['right'][0],state['right'][3],state['right'][6]= up_bottom
            state['down'][2],state['down'][1],state['down'][0]= right_left
            state['left'][8],state['left'][5],state['left'][2]= down_top
        elif direction == 'counterclockwise':
            state['up'][6],state['up'][7],state['up'][8]= right_left
            state['right'][0],state['right'][3],state['right'][6]= down_top
            state['down'][2],state['down'][1],state['down'][0]= left_right
            state['left'][8],state['left'][5],state['left'][2]= up_bottom
        elif direction == 'half_turn':
            state['up'][6],state['up'][7],state['up'][8]=down_top
            state['down'][2],state['down'][1],state['down'][0]=up_bottom
            state['right'][0],state['right'][3],state['right'][6]=left_right
            state['left'][8],state['left'][5],state['left'][2]=right_left
        return state

    def rotate_adjacent_down(self, state, direction):
        front_bottom = state['front'][6:9]
        right_bottom = state['right'][6:9]
        back_bottom  = state['back'][6:9]
        left_bottom  = state['left'][6:9]
        if direction == 'clockwise':
            state['front'][6:9]= left_bottom
            state['right'][6:9]= front_bottom
            state['back'][6:9]= right_bottom
            state['left'][6:9]= back_bottom
        elif direction == 'counterclockwise':
            state['front'][6:9]= right_bottom
            state['left'][6:9]= front_bottom
            state['back'][6:9]= left_bottom
            state['right'][6:9]= back_bottom
        elif direction == 'half_turn':
            state['front'][6:9],state['back'][6:9]= back_bottom, front_bottom
            state['right'][6:9],state['left'][6:9]= left_bottom, right_bottom
        return state

    def rotate_adjacent_left(self, state, direction):
        up_left= [state['up'][0],state['up'][3],state['up'][6]]
        front_left= [state['front'][0],state['front'][3],state['front'][6]]
        down_left= [state['down'][0],state['down'][3],state['down'][6]]
        back_right= [state['back'][8],state['back'][5],state['back'][2]]  
        if direction == 'clockwise':
            state['up'][0],state['up'][3],state['up'][6]= front_left
            state['front'][0],state['front'][3],state['front'][6]= down_left
            state['down'][0],state['down'][3],state['down'][6]= back_right
            state['back'][8],state['back'][5],state['back'][2]= up_left
        elif direction == 'counterclockwise':
            state['up'][0],state['up'][3],state['up'][6]= back_right
            state['front'][0],state['front'][3],state['front'][6]= up_left
            state['down'][0],state['down'][3],state['down'][6]= front_left
            state['back'][8],state['back'][5],state['back'][2]= down_left
        elif direction == 'half_turn':
            state['up'][0],state['up'][3],state['up'][6]= down_left
            state['down'][0],state['down'][3],state['down'][6]= up_left
            state['front'][0],state['front'][3],state['front'][6]= back_right
            state['back'][8],state['back'][5],state['back'][2]= front_left
        return state

    def rotate_adjacent_back(self, state, direction):
        up_top= [state['up'][2],state['up'][1],state['up'][0]] 
        right_right= [state['right'][8],state['right'][5],state['right'][2]]
        down_bottom= [state['down'][6],state['down'][7],state['down'][8]]  
        left_left= [state['left'][0],state['left'][3],state['left'][6]]
        if direction == 'clockwise':
            state['up'][2],state['up'][1],state['up'][0]= right_right
            state['right'][8],state['right'][5],state['right'][2]= down_bottom
            state['down'][6],state['down'][7],state['down'][8]= left_left
            state['left'][0],state['left'][3],state['left'][6]= up_top
        elif direction == 'counterclockwise':
            state['up'][2],state['up'][1],state['up'][0]= left_left
            state['right'][8],state['right'][5],state['right'][2]= up_top
            state['down'][6],state['down'][7],state['down'][8]= right_right
            state['left'][0],state['left'][3],state['left'][6]= down_bottom
        elif direction == 'half_turn':
            state['up'][2],state['up'][1],state['up'][0]= down_bottom
            state['down'][6],state['down'][7],state['down'][8]= up_top
            state['right'][8],state['right'][5],state['right'][2]= left_left
            state['left'][0],state['left'][3],state['left'][6]= right_right
        return state
    
    def enter_solving_mode(self):
        if not self.solution:
            print("No solution available. Please solve the cube first.")
            return
        self.solving_mode = True
        self.current_move_index = 0
        print("Entered solving mode. Use SPACE to navigate through the solution steps.")
        if len(self.expected_state_after_moves) > 0:
            self.current_expected_state = self.expected_state_after_moves[0]
        cv2.setWindowTitle("Rubik's Cube Solver", "Rubik's Cube Solver - Solving Mode")
        cv2.destroyWindow("Rubik's Cube Solver")
        cv2.namedWindow("Rubik's Cube Solver", cv2.WINDOW_NORMAL)
        cv2.waitKey(100)
        cv2.setWindowProperty("Rubik's Cube Solver", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def exit_solving_mode(self):
        self.solving_mode = False
        print("Exited solving mode")
        cv2.setWindowProperty("Rubik's Cube Solver", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.waitKey(100)
        cv2.setWindowTitle("Rubik's Cube Solver", "Rubik's Cube Solver")

    def draw_move_instructions(self, frame):
        if not self.solving_mode or not self.solution:
            return
        idx = self.current_move_index
        total = len(self.solution)
        if idx >= total:
            raw = "SOLVED!"
            desc = "Cube solved! All moves complete."
            display_idx = total
        else:
            raw = self.solution[idx]
            desc = self.MOVE_DESCRIPTIONS.get(raw, raw)
            display_idx = idx
        text = f"Step {min(idx+1, total)}/{total}: {desc}"
        font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (w, h), _ = cv2.getTextSize(text, font, scale, th)
        x = (frame.shape[1] - w) // 2
        y = frame.shape[0] - 40
        cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), th, cv2.LINE_AA)
        bar_h = 8
        bar_y = frame.shape[0] - 20
        progress = min(display_idx / total, 1.0) 
        cv2.rectangle(frame, (20, bar_y), (20 + int(progress * (frame.shape[1] - 40)), bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, bar_y), (frame.shape[1] - 20, bar_y + bar_h), (255, 255, 255), 1)
    
    def draw_move_visualization(self, frame, move):
        if move == "SOLVED!":
            return
        viz_width, viz_height = 180, 180
        viz_x, viz_y = frame.shape[1] - viz_width - 30, 50  
        self.draw_cube_net(frame, viz_x, viz_y, viz_width, viz_height)
        self.draw_move_arrow(frame, move, viz_x, viz_y, viz_width, viz_height)

    def draw_cube_net(self, frame, x, y, w, h):
        cell = w // 4
        layout = [
        ('up',    1, 0),
        ('left',  0, 1),
        ('front', 1, 1),
        ('right', 2, 1),
        ('back',  3, 1),
        ('down',  1, 2),
        ]
        for fname, gx, gy in layout:
            fx = x + gx*cell
            fy = y + gy*cell
            cv2.rectangle(frame, (fx, fy), (fx+cell, fy+cell),
                        self.face_display_colors[fname], 1)
            cv2.putText(frame, fname[0].upper(),
                        (fx+cell//4, fy+cell//2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.face_display_colors[fname], 1)

    def draw_move_arrow(self, frame, move, x, y, w, h):
        if not move or len(move) < 1:
            return
        face = move[0]
        dirn = '' if len(move)==1 else move[1]
        face_map = {'U': 'up', 'R': 'right', 'F': 'front', 
                'D': 'down', 'L': 'left', 'B': 'back'}
        positions = {
        'up':    (1,0), 'left':(0,1), 'front':(1,1),
        'right': (2,1), 'back':(3,1), 'down': (1,2)
        }
        cell = w // 4
        if face not in face_map or face_map[face] not in positions:
            return
        gx, gy = positions[face_map[face]]
        cx = x + gx*cell + cell//2
        cy = y + gy*cell + cell//2
        r = cell//3  
        if dirn == '':
            cv2.ellipse(frame, (cx,cy),(r,r),0,45,315,(0,255,0),1)
        elif dirn == "'":
            cv2.ellipse(frame, (cx,cy),(r,r),0,135,45,(0,255,0),1)
        else:  
            cv2.ellipse(frame, (cx,cy),(r,r),0,45,315,(0,255,0),1)
            cv2.ellipse(frame, (cx,cy),(r,r),0,135,45,(0,255,0),1)

    def draw_expected_state(self, frame):
        if not self.solving_mode or not self.current_expected_state:
            return
        cell = min(frame.shape[0], frame.shape[1]) // 20
        start_x = 40
        start_y = frame.shape[0] // 4
        layout = [
            (None,    'up',    None,   None),
            ('left',  'front', 'right', 'back'),
            (None,    'down',  None,   None)
        ]
        
        for row_idx, row in enumerate(layout):
            for col_idx, face_name in enumerate(row):
                if face_name is None:
                    continue
                x = start_x + col_idx * (cell * 3 + 10)
                y = start_y + row_idx * (cell * 3 + 10)
                cv2.rectangle(frame, (x-2, y-2), (x+cell*3+2, y+cell*3+2), 
                            self.face_display_colors.get(face_name, (200, 200, 200)), 1)
                cv2.putText(frame, face_name[0].upper(), (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                        self.face_display_colors.get(face_name, (200, 200, 200)), 1)
                if face_name in self.current_expected_state:
                    for i in range(9):
                        row, col = i // 3, i % 3
                        color_name = self.current_expected_state[face_name][i]
                        if color_name:
                            color_bgr = self.color_to_bgr(color_name)
                            cv2.rectangle(frame, 
                                        (x + col * cell, y + row * cell), 
                                        (x + (col+1) * cell, y + (row+1) * cell), 
                                        color_bgr, -1)
                            cv2.rectangle(frame, 
                                        (x + col * cell, y + row * cell), 
                                        (x + (col+1) * cell, y + (row+1) * cell), 
                                        (0, 0, 0), 1)

    def draw_solution_status(self, frame):
            if not self.solution:
                return
            font_scale = 0.5
            if self.solving_mode:
                status = f"Solution Progress: {self.current_move_index + 1}/{len(self.solution)}"
                position = (20, 30)
            else:
                if len(self.solution) > 5:
                    status = f"Solution: {' '.join(self.solution[:5])}..."
                else:
                    status = f"Solution: {' '.join(self.solution)}"
                position = (20, 20)
            cv2.putText(frame, status, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

    def next_solution_step(self):
        if not self.solving_mode or not self.solution:
            return
        if self.current_move_index < len(self.solution):
            self.current_move_index += 1
            print(f"Move {self.current_move_index}/{len(self.solution)}")
            if self.current_move_index < len(self.expected_state_after_moves):
                self.current_expected_state = self.expected_state_after_moves[self.current_move_index]
                if self.current_move_index > 0:  
                    print(f"Applied move: {self.solution[self.current_move_index-1]}")

    def previous_solution_step(self):
        if not self.solving_mode or not self.solution:
            return
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.current_expected_state = self.expected_state_after_moves[self.current_move_index]
            print(f"Move {self.current_move_index}/{len(self.solution)}: {self.solution[self.current_move_index]}")

    def reset_solution(self):
        if not self.solving_mode or not self.solution:
            return
        self.current_move_index = 0
        self.current_expected_state = self.expected_state_after_moves[0]
        print("Solution reset to the beginning")

    def exit_solving_mode(self):
        self.solving_mode = False
        print("Exited solving mode")

    def run(self):
        if not self.start_camera():
            print("Failed to start camera. Exiting.")
            return
        print("Camera started successfully.")
        print("Press '?' to toggle help text.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            self.frame = frame.copy()
            if not self.solving_mode:
                grid_points, cell_width, cell_height = self.detect_grid(frame)
                colors = self.detect_colors(frame, grid_points)
                self.draw_grid(frame, grid_points, cell_width, cell_height, colors)
                self.draw_face_indicators(frame)
            if self.show_help:
                self.draw_help(frame)
            self.draw_solution_status(frame)
            if self.solving_mode:
                self.draw_move_instructions(frame)
                self.draw_expected_state(frame)
                current_move = None
                if self.solution and self.current_move_index < len(self.solution):
                    current_move = self.solution[self.current_move_index]
                elif self.solution and self.current_move_index == len(self.solution):
                    current_move = "SOLVED!"
                if current_move:
                    self.draw_move_visualization(frame, current_move)
            
            cv2.imshow("Rubik's Cube Solver", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                self.running = False
            elif key == ord('?'):
                self.show_help = not self.show_help
            elif key == ord('c'):
                self.cube_state = {face: [''] * 9 for face in self.cube_state}
                self.faces_captured = set()
                print("Cleared all captured faces")
            elif key == ord('s'):
                self.save_cube_state()
            elif key == ord('o'):
                self.load_cube_state('cube_state.json')
            elif key == ord('z'):
                if len(self.faces_captured) >= 6:
                    success = self.solve_cube()
                    if success:
                        self.enter_solving_mode()
                else:
                    print(f"Cannot solve: only {len(self.faces_captured)} faces captured out of 6")
            elif key == 32:  
                if self.solving_mode:
                    self.next_solution_step()
                else:
                    self.capture_current_face(colors)
            elif key == ord('p'):
                if self.solving_mode:
                    self.previous_solution_step()
            elif key == 27:  
                if self.solving_mode:
                    self.exit_solving_mode()
            elif not self.solving_mode:
                if key == ord('u'):
                    self.current_face = 'up'
                    print("Selected Up face")
                elif key == ord('r'):
                    self.current_face = 'right'
                    print("Selected Right face")
                elif key == ord('f'):
                    self.current_face = 'front'
                    print("Selected Front face")
                elif key == ord('d'):
                    self.current_face = 'down'
                    print("Selected Down face")
                elif key == ord('l'):
                    self.current_face = 'left'
                    print("Selected Left face")
                elif key == ord('b'):
                    self.current_face = 'back'
                    print("Selected Back face")
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    detector = RubiksCubeDetector()
    detector.run()
if __name__ == "__main__":
        main()