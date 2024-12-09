# Server for Children's Role-Play Application

This project implements a server to support a role-playing application designed for children's educational purposes. The server integrates advanced conversational AI technology using a **Text-to-Speech (TTS) -> Language Model (LLM) -> Speech-to-Text (STT)** pipeline, providing real-time interactions that enhance early childhood education and language development.

## Features

- **Conversational AI**: Implements natural, engaging, and context-aware conversations to support role-playing scenarios.
- **Stream Responses**:
  - **LLM Streaming**: Generates and streams text responses incrementally to reduce inference latency.
  - **TTS Streaming**: Converts text responses into audio in real-time for faster feedback.
- **Integrated Pipeline**:
  - **TTS**: Fine-tuned Whisper-small model trained on datasets `한국어 아동 음성 데이터` and `자유대화 음성(소아남여, 유아 등 혼합)` from AIHub.
  - **LLM**: Utilizes OpenAI’s `gpt-4o-mini` for conversational logic.
  - **STT**: Uses OpenAI’s `tts-1` for accurate speech recognition.

## Architecture

The application is built using:

- **[FastAPI](https://fastapi.tiangolo.com/)**: A high-performance web framework for building and serving the API endpoints.
- **Docker**: Containerizes the application for consistent and efficient deployment.
- **Google Cloud Run**: Deploys the server in a scalable, managed environment for cost-effective hosting.

## Prerequisites

Before deploying or running the application, ensure you have the following:

- **Python 3.8+**
- **Docker** and **Docker Compose** installed.
- **Google Cloud Platform (GCP)** account with Cloud Run enabled.
- OpenAI API key stored as an environment variable: `OPENAI_API_KEY`.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Jaeho-Jung/Capstone2024_server.git
cd Capstone2024_server
```

### 2. Install Dependencies

Use `pip` to install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run Locally (Optional)

To run the application locally for development:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## Docker Setup

### 1. Build the Docker Image

```bash
docker build -t roleplay-server .
```

### 2. Run the Docker Container

```bash
docker run -d -p 8080:8080 --env-file .env roleplay-server
```

## Deployment

### 1. Push Docker Image to Google Container Registry (GCR)

```bash
gcloud auth configure-docker
docker tag roleplay-server gcr.io/<your-project-id>/roleplay-server
docker push gcr.io/<your-project-id>/roleplay-server
```

### 2. Deploy to Google Cloud Run

```bash
gcloud run deploy roleplay-server \
    --image gcr.io/<your-project-id>/roleplay-server \
    --platform managed \
    --region <your-region> \
    --allow-unauthenticated \
    --set-env-vars OPENAI_API_KEY=your_openai_api_key
```

## API Endpoints

### `/roleplay/reset_conversation` (POST)

Resets the conversation context to start a new dialogue.

### `/roleplay/stream` (WebSocket)

Handles streaming interactions in the following sequence:

1. Receives audio input from the client.
2. Processes the input using the **STT** pipeline.
3. Generates a response using **LLM**.
4. Converts the response to audio using **TTS**.
5. Streams the audio back to the client in real time.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

- **[FastAPI](https://fastapi.tiangolo.com/)** for powering the API.
- **OpenAI** for providing the `gpt-4o-mini` and `tts-1` models.
- **AIHub** for datasets used in fine-tuning the TTS model.
- **Google Cloud Platform** for scalable hosting through Cloud Run.
