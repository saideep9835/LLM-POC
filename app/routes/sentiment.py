from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
import os
import tempfile
import shutil
import subprocess
import time

from app.models import SentimentRequest, SentimentResponse, AudioSentimentResponse
from app.services.azure_language_service import AzureLanguageService
from app.services.document_intelligence_service import DocumentIntelligenceService
from app.services.azure_speech_service import AzureSpeechService


router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
azure_language_service = AzureLanguageService()
document_intelligence_service = DocumentIntelligenceService()
azure_speech_service = AzureSpeechService()


SUPPORTED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
}

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".ogg"}


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert input audio to WAV PCM 16kHz mono using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is required for mp3/mp4 conversion but is not installed")

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]

    try:
        start = time.monotonic()
        print(f"[audio] ffmpeg start input={input_path} output={output_path}", flush=True)
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        elapsed = time.monotonic() - start
        print(f"[audio] ffmpeg completed in {elapsed:.2f}s", flush=True)
    except subprocess.TimeoutExpired as e:
        raise ValueError("Audio conversion timed out after 120s") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "unknown ffmpeg error").strip()
        raise ValueError(f"Audio conversion to WAV failed: {err}") from e


@router.post("/text", response_model=SentimentResponse)
async def analyze_text_sentiment(request: SentimentRequest):
    """Analyze sentiment for plain text using Azure AI Language."""
    try:
        result = azure_language_service.analyze_sentiment(
            text=request.text,
            language=request.language,
            include_opinion_mining=request.include_opinion_mining,
        )
        return SentimentResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/simple", response_model=SentimentResponse)
async def analyze_text_sentiment_simple(text: str, language: str = "en"):
    """Simple query-param endpoint for quick sentiment tests."""
    try:
        result = azure_language_service.analyze_sentiment(
            text=text,
            language=language,
            include_opinion_mining=False,
        )
        return SentimentResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/document", response_model=SentimentResponse)
async def analyze_document_sentiment(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Analyze sentiment for uploaded document (PDF/DOCX/DOC/TXT)."""
    try:
        filename = file.filename or "uploaded_file"
        extension = os.path.splitext(filename.lower())[1]
        content_type = file.content_type or "application/octet-stream"

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file extension: {extension or 'unknown'}. "
                    "Supported formats are .pdf, .docx, .txt"
                ),
            )

        if content_type not in SUPPORTED_DOCUMENT_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported MIME type: {content_type}. "
                    "Supported MIME types are application/pdf, "
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain"
                ),
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        if content_type == "text/plain":
            extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()
            if not extracted_text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No readable text found in uploaded text file",
                )
        else:
            extracted_text = document_intelligence_service.extract_text(
                file_bytes=file_bytes,
            )

        result = azure_language_service.analyze_sentiment(
            text=extracted_text,
            language=language,
            include_opinion_mining=False,
        )
        return SentimentResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Document sentiment analysis failed for filename={filename}, "
                f"mime={content_type}, size_bytes={len(file_bytes) if 'file_bytes' in locals() else 0}. "
                f"Error: {str(e)}"
            ),
        )


@router.post("/audio", response_model=AudioSentimentResponse)
async def analyze_audio_sentiment(
    file: UploadFile = File(...),
    speech_language: str = Form("en-US"),
    sentiment_language: str = Form("en"),
):
    """Transcribe uploaded audio using Azure Speech, then analyze sentiment using Azure Language."""
    temp_path = None
    converted_wav_path = None
    try:
        print("[audio] /api/v1/sentiment/audio request received", flush=True)
        filename = file.filename or "uploaded_audio"
        extension = os.path.splitext(filename.lower())[1]
        print(
            f"[audio] incoming file name={filename}, ext={extension}, content_type={file.content_type}",
            flush=True,
        )
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported audio extension: {extension or 'unknown'}. "
                    "Supported formats are .wav, .mp3, .mp4, .m4a, .ogg"
                ),
            )

        file_bytes = await file.read()
        print(f"[audio] bytes_read={len(file_bytes)}", flush=True)
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        transcription_input_path = temp_path
        if extension != ".wav":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                converted_wav_path = wav_file.name
            print(f"[audio] converting to wav via ffmpeg: {temp_path} -> {converted_wav_path}", flush=True)
            _convert_to_wav(temp_path, converted_wav_path)
            transcription_input_path = converted_wav_path

        print(f"[audio] transcription starting, speech_language={speech_language}", flush=True)
        transcript = azure_speech_service.transcribe_file(
            file_path=transcription_input_path,
            language=speech_language,
        )
        print(f"[audio] transcription success, transcript_len={len(transcript)}", flush=True)

        print(f"[audio] sentiment analysis starting, sentiment_language={sentiment_language}", flush=True)
        sentiment = azure_language_service.analyze_sentiment(
            text=transcript,
            language=sentiment_language,
            include_opinion_mining=False,
        )
        print(f"[audio] sentiment analysis success, sentiment={sentiment.get('sentiment')}", flush=True)

        return AudioSentimentResponse(
            transcript=transcript,
            sentiment=sentiment["sentiment"],
            confidence_scores=sentiment["confidence_scores"],
            sentences=sentiment.get("sentences", []),
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[audio] value error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        print(f"[audio] unexpected error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio sentiment analysis failed: {str(e)}",
        )
    finally:
        print("[audio] cleanup temp files", flush=True)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_wav_path and os.path.exists(converted_wav_path):
            os.remove(converted_wav_path)
