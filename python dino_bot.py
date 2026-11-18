import time
import random
import numpy as np
import cv2
import mss
import pyautogui as pydirectinput

# ================= НАСТРОЙКИ =================
# Область игры в координатах экрана. Подгони под своё окно Chrome.
MONITOR = {"top": 300, "left": 200, "width": 1000, "height": 300}

# Координаты внутри захваченной области:
GROUND_Y = 250      # Y линии земли
DINO_X = 80         # X левого края динозавра
SCAN_AHEAD = 350    # на сколько пикселей вперёд смотрим
DARK_THRESH = 100   # порог тёмных пикселей

# Полосы поиска препятствий
CACTUS_BAND = (GROUND_Y - 45, GROUND_Y - 5)
BIRD_BAND = (GROUND_Y - 110, GROUND_Y - 50)

FPS = 30
FRAME_TIME = 1.0 / FPS

# ============================================


def get_gray(sct):
    img = np.array(sct.grab(MONITOR))[:, :, :3]  # BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def first_dark_x(gray, y1, y2, x1, x2, thresh):
    """Возвращает X первого тёмного пикселя в полосе или None."""
    h, w = gray.shape
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))

    band = gray[y1:y2, x1:x2]
    if band.size == 0:
        return None

    mask = band < thresh
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None

    return x1 + cols[0]


def detect(gray):
    # Кактус — ищем по земле
    x = first_dark_x(
        gray,
        CACTUS_BAND[0], CACTUS_BAND[1],
        DINO_X + 40, DINO_X + SCAN_AHEAD,
        DARK_THRESH
    )
    ground_dist = (x - DINO_X) if x is not None else 9999

    # Птица — ищем на высоте головы/тела
    x = first_dark_x(
        gray,
        BIRD_BAND[0], BIRD_BAND[1],
        DINO_X + 40, DINO_X + SCAN_AHEAD,
        DARK_THRESH
    )
    bird_dist = (x - DINO_X) if x is not None else 9999

    return ground_dist, bird_dist


def press_space():
    pydirectinput.press("space")


def down_on():
    pydirectinput.keyDown("down")


def down_off():
    pydirectinput.keyUp("down")


def mutate(params):
    """Немного меняем пороги, чтобы бот учился."""
    p = params.copy()
    for k in p:
        if random.random() < 0.5:
            p[k] += random.randint(-20, 20)

    p["jump_dist"] = max(80, min(300, p["jump_dist"]))
    p["duck_dist"] = max(80, min(300, p["duck_dist"]))
    return p


def main():
    params = {
        "jump_dist": 170,
        "duck_dist": 150,
    }
    best_params = params.copy()
    best_score = 0

    ducking = False
    last_jump = 0

    with mss.mss() as sct:
        print("Открой Chrome Dino. Старт через 3 секунды...")
        time.sleep(3)
        press_space()
        time.sleep(0.5)

        prev_gray = None
        static_frames = 0
        score_frames = 0
        last_time = time.time()

        while True:
            # Держим примерно 30 FPS
            now = time.time()
            dt = now - last_time
            if dt < FRAME_TIME:
                time.sleep(FRAME_TIME - dt)
            last_time = time.time()

            gray = get_gray(sct)
            ground_dist, bird_dist = detect(gray)

            # --- Приседание, если близко птица ---
            if bird_dist < params["duck_dist"]:
                if not ducking:
                    down_on()
                    ducking = True
            else:
                if ducking:
                    down_off()
                    ducking = False

            # --- Прыжок, если близко кактус ---
            if ground_dist < params["jump_dist"] and not ducking:
                if time.time() - last_jump > 0.20:
                    press_space()
                    last_jump = time.time()

            # --- Определение Game Over по остановке фона ---
            if prev_gray is not None:
                y1 = min(gray.shape[0] - 1, GROUND_Y + 2)
                y2 = min(gray.shape[0], GROUND_Y + 12)
                x1 = min(gray.shape[1] - 1, DINO_X + 200)
                x2 = min(gray.shape[1], DINO_X + 600)

                diff = cv2.absdiff(
                    gray[y1:y2, x1:x2],
                    prev_gray[y1:y2, x1:x2]
                )
                motion = diff.mean()

                if motion < 0.5:
                    static_frames += 1
                else:
                    static_frames = 0

                if static_frames > 15:  # примерно 0.5 сек без движения
                    print(f"Game Over. Score: {score_frames}, params: {params}")

                    # --- Обучение ---
                    if score_frames > best_score:
                        best_score = score_frames
                        best_params = params.copy()
                    else:
                        params = best_params.copy()

                    params = mutate(params)
                    print(f"Новые параметры: {params} | best score = {best_score}")

                    # Рестарт
                    if ducking:
                        down_off()
                        ducking = False

                    time.sleep(0.2)
                    press_space()
                    time.sleep(0.5)

                    prev_gray = None
                    static_frames = 0
                    score_frames = 0
                    continue

            prev_gray = gray
            score_frames += 1


if __name__ == "__main__":
    main()