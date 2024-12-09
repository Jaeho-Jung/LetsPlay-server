import os
from fastapi import FastAPI, APIRouter, WebSocket
from fastapi.responses import JSONResponse

from src import log
from src.gpt_service import GPTService
from src.whisper_service import WhisperService
from src.utils.utils import INSTRUCTION
api_key = os.environ.get('OPENAI_API_KEY')


app = FastAPI(title="Role-play Service")

roleplay_stream_router = APIRouter(prefix="/roleplay", tags=["Role-play Stream"])

whisper_service = WhisperService()
gpt_service = GPTService(api_key=api_key)

context = [
    {'role': 'system', 'content': INSTRUCTION}
]

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
    await websocket.accept()
    try:
        # Step 1: Receive audio data from client
        audio_data = await websocket.receive_bytes()

        # Step 2: Process audio with STT
        transcript = await whisper_service.transcribe_audio(audio_data)
        await websocket.send_json({"type": "STT", "content": transcript}) # Send transcript to client
        context.append({"role": "user", "content": f"{transcript}"}) # Update context with user input


        # Step 3: Process LLM output
        llm_response = ''
        async for sentence in gpt_service.async_generate_chat_response(context):
            await websocket.send_json({"type": "LLM", "content": sentence}) # Send LLM text to client

            # Step 4: Process TTS output
            async for audio_chunk in gpt_service.async_generate_tts_response(sentence):
                await websocket.send_bytes(audio_chunk) # Send audio chunks to client
            llm_response += ' ' + sentence

        # Update context with LLM response
        context.append({"role": "assistant", "content": f"{llm_response}"})
        
        await websocket.send_json({"type": "END"})
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

app.include_router(roleplay_stream_router)
