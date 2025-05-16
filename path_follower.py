import cv2
import numpy as np
import pyautogui
import time
import keyboard # Using 'keyboard' for global hotkey detection

# --- Configuration ---
GREEN_SQUARE_RGB = np.array([87, 175, 35]) # OpenCV uses BGR by default, so swapping R and B
GREY_PATH_RGB = np.array([72, 65, 62])   # OpenCV uses BGR by default
TOLERANCE = 20 # Increased tolerance for better color matching

SCREEN_CAPTURE_DELAY_MS = 0.5 # Delay between iterations (further reduced for maximum speed)
MOVEMENT_DELAY_S = 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

HOTKEY = 'f7'
running = False # Global state for toggling the automation

# --- Toggle Function ---s
def toggle_script():
    global running
    running = not running
    if running:
        print("Script activated. Press F7 to deactivate.")
    else:
        print("Script deactivated. Press F7 to activate.")

keyboard.add_hotkey(HOTKEY, toggle_script)

print(f"Path follower script initialized. Press {HOTKEY.upper()} to toggle automation.")

# --- Image Processing Functions ---
def capture_screen():
    """Captures the screen and converts it to an OpenCV BGR image."""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert RGB (from PIL/pyautogui) to BGR (for OpenCV)
    return frame

def find_color_mask(image, target_color_bgr, tolerance=10):
    """Creates a mask for the target color within a given tolerance."""
    lower_bound = np.clip(target_color_bgr - tolerance, 0, 255)
    upper_bound = np.clip(target_color_bgr + tolerance, 0, 255)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def find_largest_contour_center(mask):
    """Finds the largest contour in a binary mask and returns its center."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 10: # Ignore very small contours
        return None
        
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return (center_x, center_y)

# --- Movement and Keypress Functions ---
LOOK_AHEAD_DISTANCE = 30 # Pixels to look ahead for path (previous constant, might be superseded by PROBE settings)
SAMPLE_POINT_OFFSET = 15 # How far off-center to check for path width (previous constant)

# New constants for enhanced movement logic
PROBE_DISTANCE = 25       # How far from the square's center to center the probe area
PROBE_WIDTH = 30          # Width of the probe area (e.g., perpendicular to movement)
PROBE_HEIGHT = 15         # Height (depth) of the probe area (e.g., in direction of movement)
INERTIA_BONUS = 1.2       # Score multiplier for the last successful move
TURN_THRESHOLD_RATIO = 1.1 # New direction score must be this much * current direction's score to switch
MIN_PATH_PIXELS_FOR_MOVE = (PROBE_WIDTH * PROBE_HEIGHT) * 0.20 # Must have at least 20% of probe area as path

def press_key(key):
    """Presses a key and adds a small delay."""
    pyautogui.press(key)
    time.sleep(MOVEMENT_DELAY_S)

def count_path_pixels_in_rect(path_mask, rect_roi):
    """Counts white pixels (path) in a given rectangular RoI of the path_mask."""
    x, y, w, h = rect_roi
    # Ensure RoI is within mask boundaries
    mask_h, mask_w = path_mask.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(mask_w, x + w), min(mask_h, y + h)
    
    if x1 >= x2 or y1 >= y2: # RoI is outside or has no area
        return 0
        
    roi_slice = path_mask[y1:y2, x1:x2]
    return np.count_nonzero(roi_slice)

def decide_move(square_center, path_mask, last_successful_move):
    """Analyzes the path_mask around the square using probe areas to decide the next move.
    Considers inertia and requires a significantly better option to turn.
    Prevents immediate reversal of direction.
    """
    if square_center is None:
        return None

    h_mask, w_mask = path_mask.shape
    cx, cy = square_center

    # (key, dx_center, dy_center, probe_w, probe_h) - dx, dy to center of probe area
    move_probes = {
        'w': (0, -PROBE_DISTANCE, PROBE_WIDTH, PROBE_HEIGHT), # Up
        's': (0, PROBE_DISTANCE, PROBE_WIDTH, PROBE_HEIGHT),  # Down
        'a': (-PROBE_DISTANCE, 0, PROBE_HEIGHT, PROBE_WIDTH), # Left
        'd': (PROBE_DISTANCE, 0, PROBE_HEIGHT, PROBE_WIDTH),  # Right
    }
    
    opposite_moves = {
        'w': 's',
        's': 'w',
        'a': 'd',
        'd': 'a'
    }

    direction_scores = {}

    for key, (pdx, pdy, pw, ph) in move_probes.items():
        probe_center_x = cx + pdx
        probe_center_y = cy + pdy
        roi_x = probe_center_x - pw // 2
        roi_y = probe_center_y - ph // 2
        rect_roi = (roi_x, roi_y, pw, ph)
        direction_scores[key] = count_path_pixels_in_rect(path_mask, rect_roi)
        # print(f"Debug: Initial score for {key}: {direction_scores[key]}")

    # Penalize immediate reversal
    if last_successful_move:
        opposite_of_last = opposite_moves.get(last_successful_move)
        if opposite_of_last and opposite_of_last in direction_scores:
            # print(f"Debug: Penalizing {opposite_of_last} (opposite of {last_successful_move}). Original score: {direction_scores[opposite_of_last]}")
            direction_scores[opposite_of_last] = 0 # Drastically reduce score to prevent reversal
            # print(f"Debug: Penalized score for {opposite_of_last}: {direction_scores[opposite_of_last]}")

    # Apply inertia bonus (to non-penalized moves)
    if last_successful_move and last_successful_move in direction_scores:
        # Ensure we are not applying inertia to a move that was just penalized to 0 if it happened to be the last move's opposite
        # This check is a bit redundant if penalizing sets to 0, but good for clarity if penalizing strategy changes.
        if direction_scores[last_successful_move] > 0: 
            direction_scores[last_successful_move] *= INERTIA_BONUS
            # print(f"Debug: Score for {last_successful_move} after inertia: {direction_scores[last_successful_move]}")

    # Find the best direction based on scores
    best_direction = None
    max_score = -1

    # Check in a preferred order, e.g., last move first, then standard W,S,A,D
    # This can help break ties or stick with current momentum if scores are similar.
    preferred_check_order = []
    if last_successful_move:
        preferred_check_order.append(last_successful_move)
    for k in ['w', 's', 'a', 'd']:
        if k not in preferred_check_order:
            preferred_check_order.append(k)

    for key in preferred_check_order:
        score = direction_scores.get(key, 0)
        if score > max_score and score >= MIN_PATH_PIXELS_FOR_MOVE:
            max_score = score
            best_direction = key
    
    # If a best direction is found, consider if it overcomes the turn threshold 
    # relative to the (potentially inertia-boosted) last move.
    if best_direction:
        if last_successful_move and best_direction != last_successful_move:
            # Score of last move (potentially with inertia) vs score of new best (without inertia for this comparison)
            original_last_move_score = count_path_pixels_in_rect(path_mask, 
                (cx + move_probes[last_successful_move][0] - move_probes[last_successful_move][2]//2, 
                 cy + move_probes[last_successful_move][1] - move_probes[last_successful_move][3]//2, 
                 move_probes[last_successful_move][2], 
                 move_probes[last_successful_move][3]))
            
            original_best_direction_score = count_path_pixels_in_rect(path_mask, 
                (cx + move_probes[best_direction][0] - move_probes[best_direction][2]//2, 
                 cy + move_probes[best_direction][1] - move_probes[best_direction][3]//2, 
                 move_probes[best_direction][2], 
                 move_probes[best_direction][3]))

            # print(f"Debug: Checking turn: Last={last_successful_move} (orig_score={original_last_move_score}), Best={best_direction} (orig_score={original_best_direction_score})")

            # Only switch if new direction is significantly better than the original score of the current path
            if original_best_direction_score < (original_last_move_score * INERTIA_BONUS * TURN_THRESHOLD_RATIO) and original_last_move_score >= MIN_PATH_PIXELS_FOR_MOVE:
                # print(f"Debug: Turn threshold not met. Sticking with {last_successful_move}.")
                return last_successful_move # Stick with current direction due to inertia and threshold
        
        # print(f"Debug: Decided move: {best_direction} with score {max_score}")
        return best_direction

    # print("Debug: No suitable move found.")
    return None # No suitable move found

# --- Main Loop ---
def main():
    global running
    print(f"Starting main loop. Press {HOTKEY.upper()} to control automation.")
    last_successful_move = None

    try:
        while True:
            if running:
                iter_start_time = time.time()

                # 1. Capture screen and detect green square
                screen_image = capture_screen()
                green_mask = find_color_mask(screen_image, GREEN_SQUARE_RGB, TOLERANCE)
                square_center = find_largest_contour_center(green_mask)

                action_taken_this_cycle = False
                if not square_center:
                    if last_successful_move is not None: # Only print if it was previously moving
                        print("Waiting: Green square not detected...")
                    last_successful_move = None
                else:
                    # 2. If square is found, detect path and decide move
                    path_mask = find_color_mask(screen_image, GREY_PATH_RGB, TOLERANCE)
                    move = decide_move(square_center, path_mask, last_successful_move)

                    if move:
                        # print(f"Moving: {move.upper()}") # Less verbose printing for high speed
                        press_key(move) # Contains MOVEMENT_DELAY_S
                        last_successful_move = move
                        action_taken_this_cycle = True
                    else:
                        # Square found, but no clear path from its position
                        if last_successful_move is not None or not action_taken_this_cycle:
                             # Print if it was moving, or if no action yet and path unclear
                            # print("Waiting: Square detected, but no clear path found...") # User requested to remove this message
                            pass # Still pause, just silently
                        last_successful_move = None 
                
                # 3. Ensure loop rate / Cooldown period
                elapsed_time_s = time.time() - iter_start_time
                desired_cycle_time_s = SCREEN_CAPTURE_DELAY_MS / 1000.0
                
                sleep_duration = desired_cycle_time_s - elapsed_time_s
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                # If sleep_duration is <=0, the loop took longer than desired, so no explicit sleep here.
                # The MOVEMENT_DELAY_S inside press_key also contributes to cycle time when a key is pressed.

            else: # if not running
                # Sleep a bit longer when script is paused to reduce CPU usage
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Script interrupted by user (Ctrl+C).")
    finally:
        print("Cleaning up and exiting.")
        # if using cv2.imshow, uncomment this:
        # cv2.destroyAllWindows()
        keyboard.unhook_all_hotkeys() # Clean up hotkeys

if __name__ == "__main__":
    main() 