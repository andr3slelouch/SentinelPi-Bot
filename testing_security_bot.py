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
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()  # Load variables from the .env file

# --- Main Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- Mode Configuration ---
# Set TEST_MODE to True to use a local video file instead of live cameras.
TEST_MODE = True
LOCAL_VIDEO_PATH = '/home/andres/Downloads/videoplayback.mp4'  # Change to your test video path

# --- Live Mode Configuration (used if TEST_MODE is False) ---
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

# --- Global Variables & Dynamic Settings ---
bot = None
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
AUTHORIZED_CHAT_ID = None  # Will be set automatically via /start

# These dictionaries will be initialized in main() based on the operating mode
frame_queues = {}
last_face_detection_time = {}
last_motion_detection_time = {}
monitoring_enabled = {}
background_frames = {}

SETTINGS = {
    "motion_detection": True,
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
        await update.message.reply_text(f"Hello {user_name}! You are now authorized to receive security alerts.")
    elif AUTHORIZED_CHAT_ID == user_chat_id:
        await update.message.reply_text("You are already authorized.")
    else:
        await update.message.reply_text("This bot is already authorized by another user.")


async def get_settings(update, context):
    """Displays the current bot settings."""
    message = "--- Current Bot Settings ---\n"
    message += f"Motion Detection: {'Enabled' if SETTINGS['motion_detection'] else 'Disabled'}\n"
    message += f"Motion Detection Threshold: {SETTINGS['motion_threshold']}\n\n"
    if AUTHORIZED_CHAT_ID:
        message += f"Alerts are being sent to chat ID: {AUTHORIZED_CHAT_ID}\n"
    else:
        message += "No user is authorized for alerts. Send /start to authorize.\n"
    if TEST_MODE:
        message += "\n--- RUNNING IN LOCAL TEST MODE ---"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)


async def set_setting(update, context):
    """Sets a new value for a specific setting."""
    try:
        setting_name = context.args[0].lower()
        value = context.args[1].lower()

        if setting_name not in SETTINGS:
            await update.message.reply_text(f"Invalid setting. Use one of: {', '.join(SETTINGS.keys())}")
            return

        if setting_name == "motion_detection":
            if value in ["true", "on", "1", "enabled"]:
                SETTINGS[setting_name] = True
                await update.message.reply_text(f"✅ Motion Detection has been enabled.")
            elif value in ["false", "off", "0", "disabled"]:
                SETTINGS[setting_name] = False
                await update.message.reply_text(f"❌ Motion Detection has been disabled.")
            else:
                await update.message.reply_text("Invalid value. Please use 'true' or 'false'.")

        elif setting_name == "motion_threshold":
            try:
                threshold = int(value)
                if threshold > 0:
                    SETTINGS['motion_threshold'] = threshold
                    await update.message.reply_text(f"✅ Motion Threshold set to {threshold}.")
                else:
                    await update.message.reply_text("Threshold must be a positive number.")
            except ValueError:
                await update.message.reply_text("Invalid value. Please provide a number for the threshold.")

    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /set <setting_name> <value>\nExample: /set motion_detection true")


async def get_snapshot(update, context):
    """Sends a current snapshot from each active camera."""
    cam_list = ['local_test'] if TEST_MODE else list(RTSP_URLS.keys())
    for cam_name in cam_list:
        if not monitoring_enabled.get(cam_name, True):
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"Monitoring is disabled for {cam_name}.")
            continue
        if frame_queues.get(cam_name) and len(frame_queues[cam_name]) > 0:
            frame = frame_queues[cam_name][-1]
            file_path = f'{cam_name}_snapshot.jpg'
            cv2.imwrite(file_path, frame)
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(file_path, 'rb'))
            os.remove(file_path)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"No frames available from {cam_name} yet.")


# --- Alert Sending Functions ---

async def send_alert_photo(file_path, caption):
    """Sends a photo alert to the authorized user."""
    if AUTHORIZED_CHAT_ID:
        try:
            await bot.send_photo(chat_id=AUTHORIZED_CHAT_ID, photo=open(file_path, 'rb'), caption=caption)
        except Exception as e:
            print(f"Failed to send photo alert: {e}")
        finally:
            if os.path.exists(file_path): os.remove(file_path)
    else:
        print("Cannot send alert: No user has been authorized.")
        if os.path.exists(file_path): os.remove(file_path)


async def send_alert_video(file_path, caption):
    """Sends a video alert to the authorized user."""
    if AUTHORIZED_CHAT_ID:
        try:
            await bot.send_video(chat_id=AUTHORIZED_CHAT_ID, video=open(file_path, 'rb'), caption=caption)
        except Exception as e:
            print(f"Failed to send video alert: {e}")
        finally:
            if os.path.exists(file_path): os.remove(file_path)
    else:
        print("Cannot send alert: No user has been authorized.")
        if os.path.exists(file_path): os.remove(file_path)


# --- Video Processing ---

def save_and_send_post_detection_video(cam_name):
    """Saves the current buffer to a video and sends it."""
    video_path = f'{cam_name}_post_detection.mp4'
    frames_to_save = list(frame_queues[cam_name])
    if not frames_to_save: return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for frame in frames_to_save:
        out.write(frame)
    out.release()
    # Use asyncio.run() to schedule the coroutine on the main event loop
    asyncio.run(send_alert_video(video_path, f"Video clip following detection on {cam_name}"))


def video_capture_loop(cam_name, video_source):
    """
    Continuously captures and analyzes frames from a video source.
    Handles both live RTSP streams (reconnecting on failure) and local files (looping on end).
    """
    global background_frames
    print(f"Video processing thread started for: {cam_name}")
    while True:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source for {cam_name}. Retrying in 10 seconds.")
            time.sleep(10)
            continue

        print(f"Successfully opened video source for {cam_name}.")
        while True:
            ret, frame = cap.read()
            if not ret:
                if TEST_MODE:
                    print(f"End of test video file for {cam_name}. Looping...")
                else:
                    print(f"Stream disconnected for {cam_name}. Reconnecting...")
                break  # Exit inner loop to reopen the stream/file

            if not monitoring_enabled.get(cam_name, True):
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_queues[cam_name].append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_time = time.time()

            # --- Face Detection ---
            if current_time - last_face_detection_time.get(cam_name, 0) > 30:  # 30s cooldown
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    last_face_detection_time[cam_name] = current_time
                    print(f"Face detected on {cam_name}")
                    file_path = f'{cam_name}_face_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run(send_alert_photo(file_path, f"Face detected on {cam_name}!"))
                    threading.Timer(30.0, save_and_send_post_detection_video, args=[cam_name]).start()

            # --- Motion Detection ---
            if SETTINGS['motion_detection'] and (current_time - last_motion_detection_time.get(cam_name, 0) > 10):
                g_blur = cv2.GaussianBlur(gray, (21, 21), 0)
                if background_frames.get(cam_name) is None:
                    background_frames[cam_name] = g_blur
                    continue

                frame_delta = cv2.absdiff(background_frames[cam_name], g_blur)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                if thresh.sum() > SETTINGS['motion_threshold']:
                    last_motion_detection_time[cam_name] = current_time
                    print(f"Motion detected on {cam_name}")
                    file_path = f'{cam_name}_motion_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run(send_alert_photo(file_path, f"Motion detected on {cam_name}"))
                if int(current_time) % 5 == 0: background_frames[cam_name] = g_blur

            # Slow down processing in test mode to simulate real-time
            if TEST_MODE: time.sleep(1 / FPS)

        cap.release()
        time.sleep(2)  # Brief pause before attempting to reconnect/re-open


# --- Main Application ---
def main():
    """Validates configuration and starts the bot and video processing threads."""
    if not TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN not found. Please create a .env file and add your token.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    global bot
    bot = application.bot

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("settings", get_settings))
    application.add_handler(CommandHandler("set", set_setting))
    application.add_handler(CommandHandler("snapshot", get_snapshot))

    # --- Mode-specific Setup ---
    if TEST_MODE:
        print("--- RUNNING IN LOCAL TEST MODE ---")
        if not os.path.exists(LOCAL_VIDEO_PATH):
            print(f"FATAL ERROR: Test video file not found at: {LOCAL_VIDEO_PATH}")
            return
        sources = {'local_test': LOCAL_VIDEO_PATH}
    else:
        print("--- RUNNING IN LIVE RTSP MODE ---")
        sources = RTSP_URLS

    # Initialize global dictionaries for all video sources
    for name in sources.keys():
        frame_queues[name] = deque(maxlen=VIDEO_BUFFER_SIZE)
        last_face_detection_time[name] = 0
        last_motion_detection_time[name] = 0
        background_frames[name] = None
        monitoring_enabled[name] = True

    # Start one video capture thread for each source
    for name, source_path in sources.items():
        thread = threading.Thread(target=video_capture_loop, args=(name, source_path), daemon=True)
        thread.start()

    print("Bot is running... Waiting for a user to send /start to authorize alerts.")
    application.run_polling()


if __name__ == '__main__':
    main()