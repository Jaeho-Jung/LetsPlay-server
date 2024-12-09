import asyncio
from src import log
from openai import OpenAI, AsyncOpenAI


class GPTService:
    _initialized = False

    def __init__(self, api_key: str):
        """
        Initialize the GPTService class. This ensures the GPT client is only initialized once.
        
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

    def get_chat_completion(self, messages, model="gpt-4o"):
            """ 
            Retrieves a chat completion response from the language model using the provided messages.

            Args:
                messages (list): A list of dictionaries representing the conversation, each containing
                                fields such as 'role' and 'content'.
                model (str): The model to be used for generating the response. Default is 'gpt-4o'.

            Returns:
                str: The generated response content from the model.

            Raises:
                RuntimeError: If an error occurs while attempting to get the completion.
            """
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                log.error(f"Error in fetching chat completion: {str(e)}")
                raise RuntimeError("Failed to fetch chat completion.") from e

    async def generate_chat_response(self, messages):
        """
        Asynchronously generates a response from the language model using the provided messages.

        Args:
            messages (list): A list of dictionaries representing the conversation, each containing
                            fields such as 'role' and 'content'.

        Returns:
            str: The generated response content from the model.

        Raises:
            RuntimeError: If an error occurs while attempting to generate the response.
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Generating response to the following input: {messages}")
            response = await loop.run_in_executor(None, self.get_chat_completion, messages)
            log.info("Response generation completed!")
            return response
        except Exception as e:
            log.error(f"Error during asynchronous response generation: {str(e)}")
            raise RuntimeError("Failed to generate chat response.") from e
        
    async def async_generate_chat_response(self, messages, model="gpt-4o-mini", max_tokens=255, temperature=0):
            """ 
            """
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                buffer = ''
                async for data in response:
                    # Get content from the delta
                    delta = data.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        buffer += delta.content
                        # Check for sentence-ending punctuation
                        while any(punct in buffer for punct in [".", "!", "?"]):
                            # Find the first sentence-ending punctuation
                            for punct in [".", "!", "?"]:
                                if punct in buffer:
                                    sentence, buffer = buffer.split(punct, 1)
                                    sentence = sentence.strip() + punct
                                    yield sentence  # Output the complete sentence
                                    break

                # Print any remaining content in the buffer
                if buffer.strip():
                    yield buffer.strip()
            except Exception as e:
                log.error(f"Error in fetching chat completion: {str(e)}")
                raise RuntimeError("Failed to fetch chat completion.") from e
            

    async def async_generate_tts_response(self, input_text, model="tts-1", voice='nova'):
        try:
            async with self.async_client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            response_format="pcm",  # similar to WAV, but without a header chunk at the start.
            input=input_text
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    yield chunk
        except Exception as e:
            log.error(f"Error in fetching tts completion: {str(e)}")
            raise RuntimeError("Failed to fetch tts completion.") from e