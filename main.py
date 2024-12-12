import os
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src import log
from src.gpt_service import GPTService
from src.whisper_service import WhisperService
from src.utils.utils import INSTRUCTION

# Load API key from environment variables
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI(title="Role-play Service")

# Initialize services
whisper_service = WhisperService()
gpt_service = GPTService(api_key=api_key)

# Define global conversation context
context = [{'role': 'system', 'content': INSTRUCTION}]

# Role-play stream router
roleplay_stream_router = APIRouter(prefix="/roleplay", tags=["Role-play Stream"])


@roleplay_stream_router.post("/reset_conversation")
async def reset_conversation():
    """
    Reset the conversation context to start a new dialogue.
    """
    global context
    context = [{'role': 'system', 'content': INSTRUCTION}]
    log.info("Conversation context has been reset.")
    return JSONResponse(content={"message": "Conversation context reset successfully."})


@roleplay_stream_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for handling real-time role-play interactions.
    """
    await websocket.accept()
    try:
        while True:
            # Step 1: Receive audio data from the client
            audio_data = await websocket.receive_bytes()

            # Step 2: Process audio with Whisper (STT)
            transcript = await whisper_service.transcribe_audio(audio_data)
            await websocket.send_json({"type": "STT", "content": transcript})  # Send transcript to client
            context.append({"role": "user", "content": transcript})  # Update context with user input

            # Step 3: Process GPT response
            llm_response = ''
            async for sentence in gpt_service.async_generate_chat_response(context):
                await websocket.send_json({"type": "LLM", "content": sentence})  # Send LLM response to client

                # Step 4: Generate TTS audio for the LLM response
                async for audio_chunk in gpt_service.async_generate_tts_response(sentence):
                    await websocket.send_bytes(audio_chunk)  # Send audio chunk to client
                llm_response += f" {sentence}"

            # Update context with LLM response
            context.append({"role": "assistant", "content": llm_response.strip()})

            # Signal end of response processing
            await websocket.send_json({"type": "END", "content": ""})
    except WebSocketDisconnect:
        log.info("WebSocket connection closed.")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        await websocket.close()


# Include the router in the FastAPI app
app.include_router(roleplay_stream_router)