from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image
import torch
import io
import logging
import requests
from typing import Tuple
import uvicorn
from dataloader import DataPreprocessor
from models import KinshipNet_2  # Using the best model
from config import (
    PATH_CONFIG,
    INFERENCE_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kinship Recognition API",
    description="API for determining if two people in images are related",
    version="1.0.0"
)

# Initialize model and preprocessor
device = INFERENCE_CONFIG['DEVICE']
preprocessor = DataPreprocessor(is_training=False)


class ImageUrls(BaseModel):
    """Request model for image URLs"""
    image1_url: HttpUrl
    image2_url: HttpUrl


class KinshipResponse(BaseModel):
    """Response model for kinship prediction"""
    probability: float
    are_related: bool
    processing_time_ms: float
    error: str = None


class ModelManager:
    """Manages model loading and inference"""

    def __init__(self):
        self.model = None
        self.device = device

    def load_model(self):
        """Load the model if not already loaded"""
        if self.model is None:
            try:
                # Find the best model file
                model_dir = PATH_CONFIG['MODELS_DIR']
                model_files = list(model_dir.glob("best_KinshipNet_2*.pth"))
                if not model_files:
                    raise FileNotFoundError("No model checkpoint found")

                # Load the most recent best model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                checkpoint = torch.load(latest_model, map_location=self.device)

                # Initialize and load model
                self.model = KinshipNet_2().to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                logger.info(f"Model loaded successfully from {latest_model}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def get_model(self):
        """Get the loaded model, loading it if necessary"""
        if self.model is None:
            self.load_model()
        return self.model


# Initialize model manager
model_manager = ModelManager()


async def download_and_process_image(url: str) -> torch.Tensor:
    """Download image from URL and preprocess it"""
    try:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))

        # Preprocess image
        tensor = preprocessor.preprocess_image(image)
        return tensor

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image from {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict", response_model=KinshipResponse)
async def predict_kinship(image_urls: ImageUrls):
    """Predict kinship relationship between people in two images"""
    import time
    start_time = time.time()

    try:
        # Get model
        model = model_manager.get_model()

        # Download and process both images
        image1_tensor = await download_and_process_image(str(image_urls.image1_url))
        image2_tensor = await download_and_process_image(str(image_urls.image2_url))

        # Add batch dimension
        image1_tensor = image1_tensor.unsqueeze(0).to(device)
        image2_tensor = image2_tensor.unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image1_tensor, image2_tensor)
            probability = torch.sigmoid(output).item()

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return KinshipResponse(
            probability=probability,
            are_related=probability > INFERENCE_CONFIG['CLASSIFICATION_THRESHOLD'],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return KinshipResponse(
            probability=0.0,
            are_related=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Load model at startup
    model_manager.load_model()

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )




