from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
import os

from app.models import SentimentRequest, SentimentResponse
from app.services.azure_language_service import AzureLanguageService
from app.services.document_intelligence_service import DocumentIntelligenceService


router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
azure_language_service = AzureLanguageService()
document_intelligence_service = DocumentIntelligenceService()


SUPPORTED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
}

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}


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
