import asyncio
import cv2
import telegram
from telegram.ext import Application, CommandHandler
import datetime
import threading
import time
from collections import deque
import os
import yaml
from dotenv import load_dotenv

# --- Fix for Unstable RTSP Streams ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# --- Configuration Loading ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
USER_DATA_FILE = 'user_data.yml'

# --- General Settings ---
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
VIDEO_BUFFER_SIZE = 60 * FPS

# --- Global Variables ---
bot = None
main_event_loop = None
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
running_threads = {}
frame_queues = {}


# --- Data Persistence Functions ---

def load_user_data():
    if not os.path.exists(USER_DATA_FILE): return {}
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {e}")
        return {}


def save_user_data(data):
    try:
        with open(USER_DATA_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving data to YAML file: {e}")


# --- Bot Command Handlers ---

async def start(update, context):
    """Greets the user and confirms the monitoring status of their cameras."""
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name
    await update.message.reply_text(f"Hello {user_name}! Checking your camera status...")

    user_data = load_user_data()
    user_cameras = user_data.get(chat_id, {}).get('cameras', {})

    if not user_cameras:
        await update.message.reply_text("You have no cameras configured. Use /add_camera to add your first one.")
    else:
        active_cams = 0
        inactive_cams = 0
        for cam_name in user_cameras.keys():
            thread_key = f"{chat_id}-{cam_name}"
            if thread_key in running_threads and running_threads[thread_key][0].is_alive():
                active_cams += 1
            else:
                # If a thread for some reason isn't running, try to restart it.
                print(f"Found inactive camera '{cam_name}' for user {chat_id}. Attempting to restart monitoring.")
                start_monitoring_thread(chat_id, cam_name, user_cameras[cam_name])
                inactive_cams += 1

        if active_cams > 0:
            await update.message.reply_text(f"Monitoring for your {active_cams} camera(s) is already active.")
        if inactive_cams > 0:
            await update.message.reply_text(f"Just started monitoring for {inactive_cams} of your inactive camera(s).")


async def add_camera(update, context):
    """Adds a new camera URL and name for the user."""
    chat_id = update.message.chat_id
    try:
        url, name = context.args[0], context.args[1]
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
            stop_monitoring_thread(chat_id, name)
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
        thread_key = f"{chat_id}-{name}"
        # Check if the thread object exists and if the thread is actually running
        status = "✅ Active" if thread_key in running_threads and running_threads[thread_key][
            0].is_alive() else "❌ Inactive"
        message += f"- {name} ({status})\n"
    await update.message.reply_text(message)


async def get_snapshot(update, context):
    """Sends a current snapshot from a specified camera."""
    chat_id = update.message.chat_id
    try:
        cam_name = context.args[0]
        thread_key = f"{chat_id}-{cam_name}"
        if thread_key in running_threads and frame_queues[thread_key]:
            frame = frame_queues[thread_key][-1]
            file_path = f'{thread_key}_snapshot.jpg'
            cv2.imwrite(file_path, frame)
            await context.bot.send_photo(chat_id=chat_id, photo=open(file_path, 'rb'),
                                         caption=f"Snapshot from {cam_name}")
            os.remove(file_path)
        else:
            await update.message.reply_text(f"Camera '{cam_name}' is not active or has no frames yet.")
    except IndexError:
        await update.message.reply_text("Usage: /snapshot <camera_name>")


async def record_clip(update, context):
    """Records and sends a clip of the last N seconds from a specified camera."""
    chat_id = update.message.chat_id
    try:
        seconds = int(context.args[0])
        cam_name = context.args[1]
        thread_key = f"{chat_id}-{cam_name}"

        if not (0 < seconds <= 60):
            await update.message.reply_text("Duration must be between 1 and 60 seconds.")
            return

        if thread_key in running_threads and frame_queues[thread_key]:
            await update.message.reply_text(f"Generating video for the last {seconds} seconds from {cam_name}...")
            frames_to_save = list(frame_queues[thread_key])
            num_frames_to_get = seconds * FPS
            final_frames = frames_to_save[-num_frames_to_get:]

            file_path = f'{thread_key}_record.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            for frame in final_frames:
                out.write(frame)
            out.release()
            await context.bot.send_video(chat_id=chat_id, video=open(file_path, 'rb'),
                                         caption=f"Last {seconds} seconds from {cam_name}")
            os.remove(file_path)
        else:
            await update.message.reply_text(f"Camera '{cam_name}' is not active or has no frames yet.")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /record <seconds> <camera_name>")


# --- Thread and Video Processing ---

def start_monitoring_thread(chat_id, cam_name, cam_url):
    """Starts a new video capture thread for a camera if not already running."""
    thread_key = f"{chat_id}-{cam_name}"
    if thread_key in running_threads and running_threads[thread_key][0].is_alive():
        print(f"Thread for {thread_key} is already running.")
        return

    stop_event = threading.Event()
    thread = threading.Thread(target=video_capture_loop, args=(chat_id, cam_name, cam_url, stop_event), daemon=True)
    running_threads[thread_key] = (thread, stop_event)
    frame_queues[thread_key] = deque(maxlen=VIDEO_BUFFER_SIZE)
    thread.start()
    # No need to print here, it will be logged in the main startup sequence


def stop_monitoring_thread(chat_id, cam_name):
    """Signals a video capture thread to stop."""
    thread_key = f"{chat_id}-{cam_name}"
    if thread_key in running_threads:
        thread, stop_event = running_threads[thread_key]
        print(f"Stopping thread for {thread_key}...")
        stop_event.set()
        thread.join(timeout=5)
        del running_threads[thread_key]
        if thread_key in frame_queues: del frame_queues[thread_key]
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

        print(f"Successfully opened video source for {thread_key}")
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream for {thread_key} ended. Reconnecting...")
                    time.sleep(5)  # Add a small delay before reconnecting
                    break
            except Exception as e:
                print(f"Error reading frame from {thread_key}: {e}")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_queues[thread_key].append(frame)

            current_time = time.time()
            if current_time - last_detection_time > 30:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    last_detection_time = current_time
                    print(f"Face detected on {thread_key}")
                    file_path = f'{thread_key}_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run_coroutine_threadsafe(
                        send_alert_photo(chat_id, file_path, f"Face detected on camera: {cam_name}"),
                        main_event_loop
                    )

        cap.release()
    print(f"Video capture loop for {thread_key} has terminated.")


async def send_alert_photo(chat_id, file_path, caption):
    """Sends a photo alert to the specified chat_id."""
    try:
        await bot.send_photo(chat_id=chat_id, photo=open(file_path, 'rb'), caption=caption)
        print(f"Successfully sent photo alert to {chat_id}")
    except Exception as e:
        print(f"Failed to send photo to {chat_id}: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)


# --- Main Application ---
def main():
    """Validates configuration, starts all monitoring threads, and runs the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    global bot, main_event_loop
    bot = application.bot
    main_event_loop = asyncio.get_event_loop()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add_camera", add_camera))
    application.add_handler(CommandHandler("remove_camera", remove_camera))
    application.add_handler(CommandHandler("list_cameras", list_cameras))
    application.add_handler(CommandHandler("snapshot", get_snapshot))
    application.add_handler(CommandHandler("record", record_clip))

    # --- AUTOMATIC STARTUP LOGIC ---
    print("Bot starting up...")
    print("Loading existing users and cameras from user_data.yml...")
    all_user_data = load_user_data()

    if not all_user_data:
        print("No existing user data found. Bot is ready for new users.")
    else:
        active_threads = 0
        for chat_id, user_info in all_user_data.items():
            user_cameras = user_info.get('cameras', {})
            if user_cameras:
                print(f"Found {len(user_cameras)} camera(s) for user {chat_id}.")
                for cam_name, cam_url in user_cameras.items():
                    start_monitoring_thread(chat_id, cam_name, cam_url)
                    active_threads += 1
        print(f"--- Startup complete. {active_threads} monitoring thread(s) are active. ---")

    print("Bot is now polling for messages.")
    application.run_polling()


if __name__ == '__main__':
    main()