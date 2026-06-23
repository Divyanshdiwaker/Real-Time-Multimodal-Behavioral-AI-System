from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from backend.services.fusion_service import analyze_session

router = APIRouter()

@router.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    # ------------------------------------------------
    # Validate file types
    # ------------------------------------------------
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Audio file must be a .wav file"
        )

    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Image file must be jpg or png"
        )

    # ------------------------------------------------
    # Decode image
    # ------------------------------------------------
    try:
        image_bytes = await image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image — file may be corrupted"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing error: {str(e)}"
        )

    # ------------------------------------------------
    # Run analysis pipeline
    # ------------------------------------------------
    # Read audio bytes once — stream can only be read once
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        # Validate WAV header — first 4 bytes must be RIFF
        if audio_bytes[:4] != b"RIFF":
            raise HTTPException(status_code=400, detail="Audio file is not a valid WAV file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio read error: {str(e)}")

    # Run analysis pipeline
    try:
        result = await analyze_session(audio_bytes, frame)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(e)}"
        )
