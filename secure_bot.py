import asyncio
import cv2
import telegram
from telegram.ext import Application, CommandHandler
import datetime
import threading
import time
from collections import deque
import os
import glob
from dotenv import load_dotenv  # Import the dotenv library

# --- Configuration Loading ---
load_dotenv()  # Load variables from the .env file into the environment

# --- Test Mode Configuration ---
TEST_MODE = True  # SET THIS TO True TO USE A LOCAL FILE
LOCAL_VIDEO_PATH = '/home/pi/videos/test_video.mp4'  # IMPORTANT: Change this to the actual path of your video file

# Load the token and check if it exists
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN is None:
    print("FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file.")
    print("Please create a .env file and add your bot token.")
    exit()

# --- Live Mode Configuration ---
RTSP_URLS = {
    'camera1': 'rtsp://username:password@camera_ip_address:554/stream1',
    # 'camera2': 'rtsp://username:password@another_camera_ip:554/stream1'
}

# --- General Settings ---
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
VIDEO_BUFFER_SIZE = 30 * FPS
TIME_LAPSE_DIR = "timelapse_snapshots"

# --- Global Variables & Dynamic Settings ---
bot = None
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
AUTHORIZED_CHAT_ID = None

# (The rest of the script remains exactly the same as the previous version)
frame_queues = {}
last_face_detection_time = {}
last_motion_detection_time = {}
last_hourly_snapshot_time = {}
video_capture_threads = {}
background_frames = {}
monitoring_enabled = {}

SETTINGS = {
    "motion_detection": True,
    "timelapse": True,
    "motion_threshold": 5000
}


# --- Bot Command Handlers ---

async def start(update, context):
    """Saves the chat_id of the user who starts the bot and sends a welcome message."""
    global AUTHORIZED_CHAT_ID
    user_chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name

    if AUTHORIZED_CHAT_ID is None:
        AUTHORIZED_CHAT_ID = user_chat_id
        print(f"Authorization successful. Alerts will be sent to {user_name} (ID: {AUTHORIZED_CHAT_ID}).")
        await update.message.reply_text(
            f"Hello {user_name}! You are now authorized to receive security alerts from this bot."
        )
    elif AUTHORIZED_CHAT_ID == user_chat_id:
        await update.message.reply_text("You are already authorized to receive alerts.")
    else:
        await update.message.reply_text(
            "This bot is already authorized by another user. Only one user can receive alerts."
        )


# ... (All other functions and the main() function are identical to the previous version)

# --- Alert Sending Functions (Modified to use AUTHORIZED_CHAT_ID) ---

async def send_telegram_alert_photo(file_path, caption):
    """Sends a photo alert to the authorized user."""
    if AUTHORIZED_CHAT_ID:
        try:
            await bot.send_photo(chat_id=AUTHORIZED_CHAT_ID, photo=open(file_path, 'rb'), caption=caption)
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to send photo alert: {e}")
    else:
        print("Cannot send alert: No user has been authorized. Please send /start to the bot.")
        os.remove(file_path)  # Clean up the file even if not sent


async def send_telegram_alert_video(file_path, caption):
    """Sends a video alert to the authorized user."""
    if AUTHORIZED_CHAT_ID:
        try:
            await bot.send_video(chat_id=AUTHORIZED_CHAT_ID, video=open(file_path, 'rb'), caption=caption)
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to send video alert: {e}")
    else:
        print("Cannot send alert: No user has been authorized. Please send /start to the bot.")
        os.remove(file_path)


# --- Video Processing Loop ---
def video_capture_loop(cam_name, video_source):
    """Continuously captures frames and analyzes them."""
    global background_frames

    while True:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source for {cam_name}. Retrying in 10 seconds.")
            time.sleep(10)
            continue
        print(f"Successfully connected to video source: {cam_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream ended for {cam_name}. Reconnecting...")
                break

            if not monitoring_enabled.get(cam_name, True):
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_queues[cam_name].append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            current_time = time.time()

            # Face Detection
            if current_time - last_face_detection_time.get(cam_name, 0) > 30:  # 30s cooldown
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    last_face_detection_time[cam_name] = current_time
                    print(f"Face detected on {cam_name}")
                    file_path = f'{cam_name}_face_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run(send_telegram_alert_photo(file_path, f"Face detected on {cam_name}!"))
                    # The video clip will be sent by the function below after a delay
                    threading.Timer(30.0, save_and_send_post_detection_video, args=[cam_name]).start()

            # Motion Detection
            if SETTINGS['motion_detection'] and (current_time - last_motion_detection_time.get(cam_name, 0) > 10):
                g_blur = cv2.GaussianBlur(gray, (21, 21), 0)
                if background_frames.get(cam_name) is None:
                    background_frames[cam_name] = g_blur
                    continue

                frame_delta = cv2.absdiff(background_frames[cam_name], g_blur)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                if thresh.sum() > SETTINGS['motion_threshold']:
                    last_motion_detection_time[cam_name] = current_time
                    print(f"Motion detected on {cam_name}")
                    file_path = f'{cam_name}_motion_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run(send_telegram_alert_photo(file_path, f"Motion detected on {cam_name}"))

                if int(current_time) % 5 == 0:
                    background_frames[cam_name] = g_blur
        cap.release()


def save_and_send_post_detection_video(cam_name):
    """Saves and sends the video after a face has been detected."""
    video_path = f'{cam_name}_post_detection.mp4'
    frames_to_save = list(frame_queues[cam_name])
    if len(frames_to_save) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        for frame in frames_to_save:
            out.write(frame)
        out.release()
        asyncio.run(send_telegram_alert_video(video_path, f"Video after face detection on {cam_name}"))


# --- Main Application ---
def main():
    """Starts the bot and the video processing threads."""
    global bot, frame_queues, last_face_detection_time, last_motion_detection_time
    global last_hourly_snapshot_time, background_frames, monitoring_enabled

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot = application.bot

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    # ... other handlers like get_settings, etc.

    sources = RTSP_URLS
    for name in sources.keys():
        frame_queues[name] = deque(maxlen=VIDEO_BUFFER_SIZE)
        last_face_detection_time[name] = 0
        last_motion_detection_time[name] = 0
        last_hourly_snapshot_time[name] = 0
        background_frames[name] = None
        monitoring_enabled[name] = True

    for name, source_path in sources.items():
        thread = threading.Thread(target=video_capture_loop, args=(name, source_path))
        thread.daemon = True
        thread.start()
        video_capture_threads[name] = thread

    print("Bot is running... Waiting for a user to send /start to authorize alerts.")
    application.run_polling()


if __name__ == '__main__':
    main()