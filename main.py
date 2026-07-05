import os
import time
import uuid
import asyncio
from typing import Dict, Optional
from datetime import datetime
from tqdm import tqdm

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from dotenv import load_dotenv

from moviepy import VideoFileClip
from faster_whisper import WhisperModel

# Load environment variables
load_dotenv()

# Configuration
API_TOKEN = os.getenv("API_TOKEN", "default_token_please_change")
UPLOAD_DIR = "uploads"
TRANSCRIPT_DIR = "transcripts"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Application state
class TaskState:
    def __init__(self, original_filename: str):
        self.original_filename = original_filename
        self.created_at = datetime.now().isoformat()
        self.status = "queued" # queued, processing, completed, error
        self.progress = 0.0 # percentage
        self.start_time = None
        self.elapsed_time = 0.0
        self.error_message = None
        self.transcript_file = None

# Global dictionary to track tasks
tasks: Dict[str, TaskState] = {}
# Global lock to ensure only 1 file is processed at a time
task_lock = asyncio.Lock()

app = FastAPI(title="Faster-Whisper API", description="Transcription API for Railway", version="1.0.0")

# Security Dependency
def verify_token(token: str = Query(..., description="API Security Token")):
    if token != API_TOKEN:
        logger.warning(f"Unauthorized access attempt with token: {token}")
        raise HTTPException(status_code=401, detail="Invalid API token")
    return token

def extract_audio(video_path: str) -> Optional[str]:
    """Extracts audio from a video file."""
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}_temp_audio.mp3"
    
    logger.info(f"Extracting audio from video: {video_path}")
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            logger.error("No audio track found in video.")
            video.close()
            return None
        
        video.audio.write_audiofile(audio_path, logger=None)
        video.audio.close()
        video.close()
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return None

def process_transcription(task_id: str, file_path: str):
    """Background task to handle transcription."""
    logger.info(f"Task {task_id} started processing file: {file_path}")
    state = tasks[task_id]
    state.status = "processing"
    state.start_time = time.time()
    
    audio_path = None
    try:
        # Determine if it's a video and needs audio extraction
        is_video = file_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
        audio_path = file_path
        
        if is_video:
            extracted_audio = extract_audio(file_path)
            if not extracted_audio:
                raise Exception("Failed to extract audio from video.")
            audio_path = extracted_audio
            
        logger.info(f"Task {task_id}: Loading Faster-Whisper model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        logger.info(f"Task {task_id}: Starting transcription...")
        segments, info = model.transcribe(audio_path, vad_filter=True)
        total_duration = round(info.duration, 2)
        
        full_transcript = []
        current_time_stamp = 0.0
        
        # Build a beautiful, real-time progress bar based on audio timestamps
        with tqdm(
            total=total_duration,
            unit=" audio seconds",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        ) as pbar:
            for segment in segments:
                # Calculate how many seconds we moved forward
                segment_duration = segment.end - current_time_stamp
                
                # Update progress bar
                pbar.update(segment_duration)
                
                # Update state
                current_time_stamp = segment.end
                if total_duration > 0:
                    progress_percent = min(100.0, (current_time_stamp / total_duration) * 100)
                else:
                    progress_percent = 100.0
                state.progress = round(progress_percent, 2)
                state.elapsed_time = round(time.time() - state.start_time, 2)
                
                full_transcript.append(segment.text.strip())

            # Finish progress bar cleanly (catch silence at the end of the video)
            if current_time_stamp < total_duration:
                pbar.update(total_duration - current_time_stamp)
            
        transcript_text = " ".join(full_transcript)
        
        # Save transcript with datetime suffix for easy identification
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Extract real base name if it starts with a uuid (e.g. uuid_filename)
        if "_" in base_name:
            real_base_name = base_name.split("_", 1)[-1]
        else:
            real_base_name = base_name
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt = os.path.join(TRANSCRIPT_DIR, f"{real_base_name}_{timestamp}_transcript.txt")
        
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(transcript_text)
            
        # Update final state
        state.status = "completed"
        state.progress = 100.0
        state.elapsed_time = round(time.time() - state.start_time, 2)
        state.transcript_file = output_txt
        logger.success(f"Task {task_id}: Transcription completed successfully in {state.elapsed_time}s.")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        state.status = "error"
        state.error_message = str(e)
    finally:
        # Cleanup media files (both uploaded and extracted temp audio)
        logger.info(f"Task {task_id}: Cleaning up media files...")
        if os.path.exists(file_path):
            os.remove(file_path)
        if audio_path and audio_path != file_path and os.path.exists(audio_path):
            os.remove(audio_path)

async def queue_transcription(task_id: str, file_path: str):
    """Wrapper to acquire lock and run blocking process in an executor."""
    async with task_lock:
        # We run the blocking process in a threadpool so it doesn't block the main event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_transcription, task_id, file_path)

@app.post("/api/v1/transcribe", tags=["Transcription"])
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    token: str = Depends(verify_token)
):
    """
    Upload an audio or video file for transcription. 
    Returns a task_id to check progress.
    """
    task_id = str(uuid.uuid4())
    secure_filename = os.path.basename(file.filename) if file.filename else "upload.tmp"
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{secure_filename}")
    
    # Save the uploaded file
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")
    
    # Initialize task state
    tasks[task_id] = TaskState(original_filename=secure_filename)
    
    # Queue the background task
    background_tasks.add_task(queue_transcription, task_id, file_path)
    
    logger.info(f"Task {task_id} created and queued for file: {file.filename}")
    return {"task_id": task_id, "status": "queued", "message": "File uploaded and queued for processing."}

@app.get("/api/v1/tasks", tags=["Transcription"])
async def list_tasks(token: str = Depends(verify_token)):
    """
    List all tasks, their original filenames, and current status.
    """
    task_list = []
    for t_id, state in tasks.items():
        task_list.append({
            "task_id": t_id,
            "original_filename": state.original_filename,
            "created_at": state.created_at,
            "status": state.status,
            "progress": state.progress,
            "elapsed_time": state.elapsed_time
        })
    return {"tasks": task_list}

@app.get("/api/v1/progress/{task_id}", tags=["Transcription"])
async def get_progress(task_id: str, token: str = Depends(verify_token)):
    """
    Get the progress of a transcription task.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    state = tasks[task_id]
    
    return {
        "task_id": task_id,
        "status": state.status,
        "progress": state.progress,
        "elapsed_time": state.elapsed_time,
        "error_message": state.error_message
    }

@app.get("/api/v1/download/{task_id}", tags=["Transcription"])
async def download_transcript(task_id: str, token: str = Depends(verify_token)):
    """
    Download the completed transcript.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    state = tasks[task_id]
    
    if state.status != "completed":
        raise HTTPException(status_code=400, detail=f"Transcript not ready. Current status: {state.status}")
        
    if not state.transcript_file or not os.path.exists(state.transcript_file):
        raise HTTPException(status_code=404, detail="Transcript file not found on disk.")
        
    return FileResponse(
        path=state.transcript_file,
        filename=os.path.basename(state.transcript_file),
        media_type="text/plain"
    )
