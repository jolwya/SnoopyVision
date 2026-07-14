import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

ASSET_DIR = "assets"
RUNNING_IMG_PATH = os.path.join(ASSET_DIR, "snoopy_running.png")
CAUGHT_IMG_PATH = os.path.join(ASSET_DIR, "snoopy_caught.png")

SPRITE_SIZE = 120 # width/height sprite scale
CATCH_RADIUS = 60 # closenes of fingertip
SPEED = 15 # pixels per frame of sprite
CAUGHT_DISPLAY_TIME = 1.0 # seconds to show the "caught" pose 

# resizeable feature
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

cv2.namedWindow("SnoopyVision by Julia", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SnoopyVision by Julia", frame_w, frame_h)

# functions
def load_sprite(path, target_size):
    if not os.path.exists(path):
        print(f"Missing asset: {path}! Will be using placeholder instead.")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        print(f"{path} has no alpha channel! Will be using placeholder instead.")
        return None

    h, w = img.shape[:2]
    scale = target_size / max(h, w) # fit longest side to target size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_placeholder(frame, center, size, color):
    cv2.circle(frame, center, size // 2, color, -1)

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h: # clipping for if sprite goes offscreen
        x = max(0, min(x, bg_w - w))
        y = max(0, min(y, bg_h - h))

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    return background


# setup hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

running_sprite = load_sprite(RUNNING_IMG_PATH, SPRITE_SIZE)
caught_sprite = load_sprite(CAUGHT_IMG_PATH, SPRITE_SIZE)

white_connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
white_dot_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
red_dot_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3) # red catching fingertip dot

landmark_style = {
    idx: (red_dot_style if idx == mp_hands.HandLandmark.INDEX_FINGER_TIP else white_dot_style)
    for idx in range(21)
}

def sprite_dims(sprite):
    if sprite is not None:
        h, w = sprite.shape[:2]
        return w, h
    return SPRITE_SIZE, SPRITE_SIZE

# snoopy state
cur_w, cur_h = sprite_dims(running_sprite)
snoopy_x, snoopy_y = random.randint(0, frame_w - cur_w), random.randint(0, frame_h - cur_h)
vel_x, vel_y = random.choice([-SPEED, SPEED]), random.choice([-SPEED, SPEED])
is_caught = False
caught_time = 0
score = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # snoopy catch logic
    fingertip = None
    if results.multi_hand_landmarks:
        # hand skeleton
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                white_connection_style,
            )
            lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            fingertip = (int(lm.x * frame_w), int(lm.y * frame_h))

    sprite = caught_sprite if is_caught else running_sprite 
    cur_w, cur_h = sprite_dims(sprite)

    if not is_caught:
        snoopy_x += vel_x
        snoopy_y += vel_y

        if snoopy_x <= 0 or snoopy_x + cur_w >= frame_w:
            vel_x *= -1
        if snoopy_y <= 0 or snoopy_y + cur_h >= frame_h:
            vel_y *= -1

        snoopy_x = max(0, min(snoopy_x, frame_w - cur_w))
        snoopy_y = max(0, min(snoopy_y, frame_h - cur_h))

        if fingertip is not None:
            snoopy_center = (snoopy_x + cur_w // 2, snoopy_y + cur_h // 2)
            dist = np.hypot(fingertip[0] - snoopy_center[0], fingertip[1] - snoopy_center[1])
            if dist < CATCH_RADIUS:
                is_caught = True
                caught_time = time.time()
                score += 1
    else:
        if time.time() - caught_time > CAUGHT_DISPLAY_TIME:
            is_caught = False
            new_w, new_h = sprite_dims(running_sprite)
            snoopy_x = random.randint(0, frame_w - new_w)
            snoopy_y = random.randint(0, frame_h - new_h)
            vel_x, vel_y = random.choice([-SPEED, SPEED]), random.choice([-SPEED, SPEED])

    if sprite is not None:
        frame = overlay_png(frame, sprite, snoopy_x, snoopy_y)
    else:
        color = (0, 0, 255) if is_caught else (0, 255, 0)
        center = (snoopy_x + cur_w // 2, snoopy_y + cur_h // 2)
        draw_placeholder(frame, center, SPRITE_SIZE, color)

    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # score display

    cv2.imshow("SnoopyVision by Julia", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
