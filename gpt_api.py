import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, WebSocket
from fastapi.responses import JSONResponse

from src import log
from src.whisper_service import WhisperService
from src.gpt_service import GPTService
from src.utils.utils import INSTRUCTION


api_key = os.environ.get('OPENAI_API_KEY')

app = FastAPI(title="Chat Service")

roleplay_router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize Whisper and GPT services
whisper_service = WhisperService()
gpt_service = GPTService(api_key=api_key)

# Store conversation history
context = [
    {'role': 'system', 'content': INSTRUCTION}
]

@roleplay_router.post("/reset_conversation")
async def reset_conversation():
    """
    Reset the conversation context to start a new dialogue.
    """
    global context
    context = [{'role': 'system', 'content': INSTRUCTION}]
    log.info("Conversation context has been reset.")
    return JSONResponse(content={"message": "Conversation context reset successfully."})


@roleplay_router.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process an audio file, transcribe it using WhisperService, and generate a response using GPTService.

    Args:
        file (UploadFile): The uploaded audio file to be processed.

    Returns:
        JSONResponse: The transcription of the audio and the generated response.
    """
    # Create a temporary .WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        log.info("Temporary file creation")
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
        log.info(f"Temporary file {temp_file_path} created")

    # Initialize time measurements
    transcription_time = 0
    gpt_response_time = 0
    overall_start_time = time.time()

    try:
        # Transcribe the audio using WhisperService
        transcription_start_time = time.time()
        transcription = await whisper_service.transcribe_audio(temp_file_path)
        transcription_end_time = time.time()
        transcription_time = transcription_end_time - transcription_start_time
        log.info("Transcription completed")

        # Update conversation history with user input
        context.append({"role": "user", "content": f"{transcription}"})

        # Generate a response using GPTService
        gpt_response_start_time = time.time()
        gpt_response = await gpt_service.generate_chat_response(context)
        gpt_response_end_time = time.time()
        gpt_response_time = gpt_response_end_time - gpt_response_start_time

        # Update conversation history with GPT response
        context.append({"role": "assistant", "content": f"{gpt_response}"})

        overall_end_time = time.time()
        overall_processing_time = overall_end_time - overall_start_time

        return JSONResponse(content={
            "transcription": transcription,
            "response": gpt_response,
            "times": {
                "transcription_time": transcription_time,
                "gpt_response_time": gpt_response_time,
                "overall_processing_time": overall_processing_time
            }
        })
    except Exception as e:
        log.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        log.info(f"Removing temporary file {temp_file_path}")
        os.remove(temp_file_path)

app.include_router(roleplay_router)