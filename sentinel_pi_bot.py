import asyncio
import cv2
import telegram
from telegram.ext import Application, CommandHandler
import datetime
import threading
import time
from collections import deque
import os
import yaml  # Import the YAML library
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
USER_DATA_FILE = 'user_data.yml'

# --- General Settings ---
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
VIDEO_BUFFER_SIZE = 30 * FPS

# --- Global Variables ---
bot = None
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
running_threads = {}  # Manages active camera threads { 'chat_id-cam_name': (thread, stop_event) }
frame_queues = {}  # { 'chat_id-cam_name': deque() }


# --- Data Persistence Functions ---

def load_user_data():
    """Loads user and camera data from the YAML file."""
    if not os.path.exists(USER_DATA_FILE):
        return {}
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {e}")
        return {}


def save_user_data(data):
    """Saves the given data to the YAML file."""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving data to YAML file: {e}")


# --- Bot Command Handlers ---

async def start(update, context):
    """Greets the user and starts monitoring their saved cameras."""
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name

    await update.message.reply_text(f"Hello {user_name}! Checking for your saved cameras...")

    user_data = load_user_data()
    user_cameras = user_data.get(chat_id, {}).get('cameras', {})

    if not user_cameras:
        await update.message.reply_text("You have no cameras configured. Use /add_camera to add one.")
    else:
        await update.message.reply_text(f"Found {len(user_cameras)} cameras. Starting monitoring...")
        for cam_name, cam_url in user_cameras.items():
            start_monitoring_thread(chat_id, cam_name, cam_url)

        await update.message.reply_text("Monitoring has started.")


async def add_camera(update, context):
    """Adds a new camera URL and name for the user."""
    chat_id = update.message.chat_id
    try:
        url = context.args[0]
        name = context.args[1]

        if ' ' in name:
            await update.message.reply_text("Camera name cannot contain spaces.")
            return

        user_data = load_user_data()
        if chat_id not in user_data:
            user_data[chat_id] = {'cameras': {}}

        user_data[chat_id]['cameras'][name] = url
        save_user_data(user_data)

        start_monitoring_thread(chat_id, name, url)
        await update.message.reply_text(f"Camera '{name}' added and monitoring has started.")

    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /add_camera <rtsp_url> <camera_name>")


async def remove_camera(update, context):
    """Removes a camera and stops its monitoring thread."""
    chat_id = update.message.chat_id
    try:
        name = context.args[0]
        user_data = load_user_data()

        if chat_id in user_data and name in user_data[chat_id]['cameras']:
            # Stop the thread
            stop_monitoring_thread(chat_id, name)

            # Remove from data and save
            del user_data[chat_id]['cameras'][name]
            save_user_data(user_data)

            await update.message.reply_text(f"Camera '{name}' has been removed.")
        else:
            await update.message.reply_text(f"Camera '{name}' not found.")

    except IndexError:
        await update.message.reply_text("Usage: /remove_camera <camera_name>")


async def list_cameras(update, context):
    """Lists all configured cameras for the user."""
    chat_id = update.message.chat_id
    user_data = load_user_data()
    user_cameras = user_data.get(chat_id, {}).get('cameras', {})

    if not user_cameras:
        await update.message.reply_text("You have no cameras configured.")
        return

    message = "Your configured cameras:\n"
    for name, url in user_cameras.items():
        status = "Active" if f"{chat_id}-{name}" in running_threads else "Inactive"
        message += f"- {name} ({status})\n"

    await update.message.reply_text(message)


# --- Thread and Video Processing ---

def start_monitoring_thread(chat_id, cam_name, cam_url):
    """Starts a new video capture thread for a camera if not already running."""
    thread_key = f"{chat_id}-{cam_name}"
    if thread_key in running_threads:
        print(f"Thread for {thread_key} is already running.")
        return

    stop_event = threading.Event()
    thread = threading.Thread(target=video_capture_loop, args=(chat_id, cam_name, cam_url, stop_event), daemon=True)

    running_threads[thread_key] = (thread, stop_event)
    frame_queues[thread_key] = deque(maxlen=VIDEO_BUFFER_SIZE)

    thread.start()
    print(f"Started monitoring thread for {thread_key}")


def stop_monitoring_thread(chat_id, cam_name):
    """Signals a video capture thread to stop."""
    thread_key = f"{chat_id}-{cam_name}"
    if thread_key in running_threads:
        thread, stop_event = running_threads[thread_key]
        print(f"Stopping thread for {thread_key}...")
        stop_event.set()
        thread.join(timeout=5)  # Wait for the thread to finish
        del running_threads[thread_key]
        del frame_queues[thread_key]
        print(f"Thread for {thread_key} stopped.")


def video_capture_loop(chat_id, cam_name, cam_url, stop_event):
    """Continuously captures and analyzes frames until stop_event is set."""
    thread_key = f"{chat_id}-{cam_name}"
    last_detection_time = 0

    while not stop_event.is_set():
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            print(f"Error opening stream for {thread_key}. Retrying in 20s.")
            time.sleep(20)
            continue

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Stream disconnected for {thread_key}. Reconnecting...")
                break  # Exit inner loop to try reopening the stream

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_queues[thread_key].append(frame)

            # --- Face Detection ---
            current_time = time.time()
            if current_time - last_detection_time > 30:  # 30s cooldown
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    last_detection_time = current_time
                    print(f"Face detected on {thread_key}")
                    file_path = f'{thread_key}_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run(send_alert_photo(chat_id, file_path, f"Face detected on camera: {cam_name}"))

        cap.release()

    print(f"Video capture loop for {thread_key} has terminated.")


# --- Alert Sending Functions ---

async def send_alert_photo(chat_id, file_path, caption):
    """Sends a photo alert to the specified chat_id."""
    try:
        await bot.send_photo(chat_id=chat_id, photo=open(file_path, 'rb'), caption=caption)
    except Exception as e:
        print(f"Failed to send photo to {chat_id}: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# --- Main Application ---
def main():
    """Validates configuration and starts the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    global bot
    bot = application.bot

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add_camera", add_camera))
    application.add_handler(CommandHandler("remove_camera", remove_camera))
    application.add_handler(CommandHandler("list_cameras", list_cameras))

    print("Bot is running... Send /start to begin.")
    application.run_polling()


if __name__ == '__main__':
    main()