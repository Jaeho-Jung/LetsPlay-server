import asyncio
from src import log
from openai import OpenAI, AsyncOpenAI


class GPTService:
    """
    A service for interacting with OpenAI's GPT-based APIs.
    Provides synchronous and asynchronous methods for generating responses from chat models and TTS models.
    """
    _initialized = False

    def __init__(self, api_key: str):
        """
        Initialize the GPTService instance. Ensures the GPT client is initialized only once.

        Args:
            api_key (str): API key for authenticating with the OpenAI service.
        """
        if not GPTService._initialized:
            try:
                log.info("Starting GPT service...")
                self.client = OpenAI(api_key=api_key)
                self.async_client = AsyncOpenAI(api_key=api_key)
                GPTService._initialized = True
                log.info("GPT service started successfully!")
            except Exception as e:
                log.error(f"Error during GPT service initialization: {str(e)}")
                raise

    def get_chat_completion(self, messages: list, model: str = "gpt-4o") -> str:
        """
        Synchronously retrieves a chat completion response from the language model.

        Args:
            messages (list): List of dictionaries representing the conversation, each with 'role' and 'content'.
            model (str): Model name to use for generating the response (default: 'gpt-4o').

        Returns:
            str: The response content from the model.

        Raises:
            RuntimeError: If an error occurs while fetching the response.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"Error fetching chat completion: {str(e)}")
            raise RuntimeError("Failed to fetch chat completion.") from e

    async def generate_chat_response(self, messages: list) -> str:
        """
        Asynchronously generates a response from the language model.

        Args:
            messages (list): List of dictionaries representing the conversation.

        Returns:
            str: The response content from the model.

        Raises:
            RuntimeError: If an error occurs while generating the response.
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Generating response for input: {messages}")
            response = await loop.run_in_executor(None, self.get_chat_completion, messages)
            log.info("Response generation completed!")
            return response
        except Exception as e:
            log.error(f"Error during asynchronous response generation: {str(e)}")
            raise RuntimeError("Failed to generate chat response.") from e

    async def async_generate_chat_response(self, messages: list, model: str = "gpt-4o-mini", max_tokens: int = 255, temperature: float = 0):
        """
        Asynchronously streams chat responses from the language model.

        Args:
            messages (list): List of dictionaries representing the conversation.
            model (str): Model name (default: 'gpt-4o-mini').
            max_tokens (int): Maximum number of tokens in the response.
            temperature (float): Sampling temperature for response generation.

        Yields:
            str: Sentences as they are generated.

        Raises:
            RuntimeError: If an error occurs while fetching the response.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            buffer = ''
            async for data in response:
                delta = data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    buffer += delta.content
                    while any(punct in buffer for punct in [".", "!", "?"]):
                        for punct in [".", "!", "?"]:
                            if punct in buffer:
                                sentence, buffer = buffer.split(punct, 1)
                                yield sentence.strip() + punct
                                break

            if buffer.strip():
                yield buffer.strip()
        except Exception as e:
            log.error(f"Error in streaming chat completion: {str(e)}")
            raise RuntimeError("Failed to fetch chat completion.") from e

    async def async_generate_tts_response(self, input_text: str, model: str = "tts-1", voice: str = 'nova'):
        """
        Asynchronously generates TTS (Text-to-Speech) audio from the input text.

        Args:
            input_text (str): Text to be converted to speech.
            model (str): TTS model name (default: 'tts-1').
            voice (str): Voice configuration for the TTS output (default: 'nova').

        Yields:
            bytes: Audio data chunks.

        Raises:
            RuntimeError: If an error occurs while generating the TTS response.
        """
        try:
            async with self.async_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                response_format="pcm",
                input=input_text
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    yield chunk
        except Exception as e:
            log.error(f"Error in TTS generation: {str(e)}")
            raise RuntimeError("Failed to fetch TTS response.") from e