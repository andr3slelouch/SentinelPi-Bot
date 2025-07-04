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

# --- !! MODE CONFIGURATION !! ---
TEST_MODE = False
LOCAL_VIDEO_PATH = "/home/andres/Downloads/videoplayback.mp4"

# --- Model Configuration ---
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
YUNET_CONFIDENCE_THRESHOLD = 0.6

# --- General Settings ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
VIDEO_BUFFER_SIZE = 60 * FPS

# --- Global Variables ---
bot = None
main_event_loop = None
face_detector = None
running_threads = {}
frame_queues = {}
AUTHORIZED_CHAT_ID = None


# (The code for data persistence and command handlers is unchanged)
# --- Data Persistence Functions (Unchanged) ---
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


# --- Bot Command Handlers (Unchanged) ---
async def start(update, context):
    global AUTHORIZED_CHAT_ID
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name
    if AUTHORIZED_CHAT_ID is None:
        AUTHORIZED_CHAT_ID = chat_id
        print(f"Authorization successful. Alerts will be sent to {user_name} (ID: {AUTHORIZED_CHAT_ID}).")
        await update.message.reply_text(f"Hello {user_name}! You are now authorized to receive all security alerts.")
    elif AUTHORIZED_CHAT_ID == chat_id:
        await update.message.reply_text("You are already the authorized user.")
    else:
        await update.message.reply_text("This bot is already authorized by another user.")
        return

    if TEST_MODE:
        await update.message.reply_text("Bot is in Test Mode, monitoring the local video file.")
    else:
        user_data = load_user_data()
        user_cameras = user_data.get(chat_id, {}).get('cameras', {})
        if user_cameras:
            num_active = sum(1 for cam_name in user_cameras if f"{chat_id}-{cam_name}" in running_threads)
            await update.message.reply_text(f"Monitoring for your {num_active} camera(s) is active.")
        else:
            await update.message.reply_text("You have no live cameras configured. Use /add_camera to add one.")


async def add_camera(update, context):
    if TEST_MODE:
        await update.message.reply_text("This command is disabled in Test Mode.")
        return
    chat_id = update.message.chat_id
    try:
        url, name = context.args[0], context.args[1]
        if ' ' in name:
            await update.message.reply_text("Camera name cannot contain spaces.")
            return
        user_data = load_user_data()
        if chat_id not in user_data: user_data[chat_id] = {'cameras': {}}
        user_data[chat_id]['cameras'][name] = url
        save_user_data(user_data)
        start_monitoring_thread(name, url, f"{chat_id}-{name}")
        await update.message.reply_text(f"Camera '{name}' added and monitoring has started.")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /add_camera <rtsp_url> <camera_name>")


async def remove_camera(update, context):
    if TEST_MODE:
        await update.message.reply_text("This command is disabled in Test Mode.")
        return
    chat_id = update.message.chat_id
    try:
        name = context.args[0]
        user_data = load_user_data()
        if chat_id in user_data and name in user_data[chat_id]['cameras']:
            stop_monitoring_thread(f"{chat_id}-{name}")
            del user_data[chat_id]['cameras'][name]
            save_user_data(user_data)
            await update.message.reply_text(f"Camera '{name}' has been removed.")
        else:
            await update.message.reply_text(f"Camera '{name}' not found.")
    except IndexError:
        await update.message.reply_text("Usage: /remove_camera <camera_name>")


async def list_cameras(update, context):
    if TEST_MODE:
        await update.message.reply_text("Running in Test Mode with video:\n- test-cam (✅ Active)")
        return
    chat_id = update.message.chat_id
    user_data = load_user_data()
    user_cameras = user_data.get(chat_id, {}).get('cameras', {})
    if not user_cameras:
        await update.message.reply_text("You have no cameras configured.")
        return
    message = "Your configured cameras:\n"
    for name, url in user_cameras.items():
        thread_key = f"{chat_id}-{name}"
        status = "✅ Active" if thread_key in running_threads and running_threads[thread_key][
            0].is_alive() else "❌ Inactive"
        message += f"- {name} ({status})\n"
    await update.message.reply_text(message)


async def get_snapshot(update, context):
    try:
        cam_name = context.args[0]
        chat_id = update.message.chat_id
        thread_key = "test-cam" if TEST_MODE else f"{chat_id}-{cam_name}"
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
        usage = "/snapshot test-cam" if TEST_MODE else "/snapshot <camera_name>"
        await update.message.reply_text(f"Usage: {usage}")


async def record_clip(update, context):
    try:
        seconds = int(context.args[0])
        cam_name = context.args[1]
        chat_id = update.message.chat_id
        thread_key = "test-cam" if TEST_MODE else f"{chat_id}-{cam_name}"
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
            for frame in final_frames: out.write(frame)
            out.release()
            await context.bot.send_video(chat_id=chat_id, video=open(file_path, 'rb'),
                                         caption=f"Last {seconds} seconds from {cam_name}")
            os.remove(file_path)
        else:
            await update.message.reply_text(f"Camera '{cam_name}' is not active or has no frames yet.")
    except (IndexError, ValueError):
        usage = "/record <seconds> test-cam" if TEST_MODE else "/record <seconds> <camera_name>"
        await update.message.reply_text(f"Usage: {usage}")


def start_monitoring_thread(cam_name, cam_url, thread_key):
    if thread_key in running_threads and running_threads[thread_key][0].is_alive():
        return
    stop_event = threading.Event()
    thread = threading.Thread(target=video_capture_loop, args=(cam_name, cam_url, thread_key, stop_event), daemon=True)
    running_threads[thread_key] = (thread, stop_event)
    frame_queues[thread_key] = deque(maxlen=VIDEO_BUFFER_SIZE)
    thread.start()


def stop_monitoring_thread(thread_key):
    if thread_key in running_threads:
        thread, stop_event = running_threads[thread_key]
        print(f"Stopping thread for {thread_key}...")
        stop_event.set()
        thread.join(timeout=5)
        del running_threads[thread_key]
        if thread_key in frame_queues: del frame_queues[thread_key]
        print(f"Thread for {thread_key} stopped.")


def video_capture_loop(cam_name, cam_url, thread_key, stop_event):
    last_detection_time = 0
    while not stop_event.is_set():
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            print(f"Error opening source for {thread_key}. Retrying in 20s.")
            time.sleep(20)
            continue
        print(f"Successfully opened video source for {thread_key}")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if TEST_MODE:
                    print(f"End of test video file. Looping...")
                else:
                    print(f"Stream for {thread_key} ended. Reconnecting...")
                time.sleep(2)
                break

            # This is the critical fix for the cv2.error crash
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            current_time = time.time()
            if current_time - last_detection_time > 5:
                status, faces = face_detector.detect(frame)
                if faces is not None:
                    last_detection_time = current_time
                    print(f"YuNet face detected on {thread_key}")
                    file_path = f'{thread_key}_detected.jpg'
                    cv2.imwrite(file_path, frame)
                    asyncio.run_coroutine_threadsafe(
                        send_alert_photo(file_path, f"Face detected on camera: {cam_name}"),
                        main_event_loop
                    )

            frame_queues[thread_key].append(frame)
            if TEST_MODE: time.sleep(1 / FPS)

        cap.release()
    print(f"Video capture loop for {thread_key} has terminated.")


async def send_alert_photo(file_path, caption):
    if AUTHORIZED_CHAT_ID:
        try:
            await bot.send_photo(chat_id=AUTHORIZED_CHAT_ID, photo=open(file_path, 'rb'), caption=caption)
            print(f"Successfully sent photo alert to {AUTHORIZED_CHAT_ID}")
        except Exception as e:
            print(f"Failed to send photo: {e}")
        finally:
            if os.path.exists(file_path): os.remove(file_path)
    else:
        print("Cannot send alert: No user has been authorized via /start.")
        if os.path.exists(file_path): os.remove(file_path)


# --- Main Application ---
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    global face_detector
    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"FATAL ERROR: YuNet model file not found at '{YUNET_MODEL_PATH}'")
        return

    face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (FRAME_WIDTH, FRAME_HEIGHT),
                                              YUNET_CONFIDENCE_THRESHOLD)
    print("YuNet face detector model loaded successfully.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    global bot, main_event_loop
    bot = application.bot

    # FIX: Reverted to asyncio.get_event_loop(). This will show a non-fatal
    # DeprecationWarning but it works correctly and avoids the AttributeError crash.
    main_event_loop = asyncio.get_event_loop()

    handlers = [
        CommandHandler("start", start), CommandHandler("add_camera", add_camera),
        CommandHandler("remove_camera", remove_camera), CommandHandler("list_cameras", list_cameras),
        CommandHandler("snapshot", get_snapshot), CommandHandler("record", record_clip)
    ]
    for handler in handlers: application.add_handler(handler)

    print("Bot starting up...")
    if TEST_MODE:
        print("=" * 25 + "\n--- RUNNING IN TEST MODE ---\n" + "=" * 25)
        if not os.path.exists(LOCAL_VIDEO_PATH):
            print(f"FATAL ERROR: Test video file not found at '{LOCAL_VIDEO_PATH}'")
            return
        start_monitoring_thread("test-cam", LOCAL_VIDEO_PATH, "test-cam")
        print("Test mode monitoring thread started.")
    else:
        print("--- RUNNING IN LIVE MODE ---")
        all_user_data = load_user_data()
        if all_user_data:
            active_threads = 0
            for chat_id, user_info in all_user_data.items():
                for cam_name, cam_url in user_info.get('cameras', {}).items():
                    start_monitoring_thread(cam_name, cam_url, f"{chat_id}-{cam_name}")
                    active_threads += 1
            print(f"--- Startup complete. {active_threads} monitoring thread(s) are active. ---")
        else:
            print("No existing user data found. Bot is ready for new users.")

    print("Bot is now polling for messages. Send /start to authorize alerts.")
    application.run_polling()


if __name__ == '__main__':
    main()