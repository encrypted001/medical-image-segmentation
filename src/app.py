"""Streamlit interface for interactive U-Net segmentation predictions."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.dataset import SUPPORTED_EXTENSIONS
from src.inference import load_model, predict_image
from src.visualize import compose_visualization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
st.set_page_config(page_title="MRI Segmentation Demo", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading the saved U-Net checkpoint...")
def _cached_model(checkpoint_path: str, device_name: str):
    """Cache model weights between Streamlit reruns; never train here."""
    return load_model(Path(checkpoint_path), torch.device(device_name))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    st.title("MRI Segmentation Demo")
    st.write("Upload an MRI image to generate a binary segmentation mask with the saved U-Net.")
    st.info(
        "Educational research prototype only. This software is not a medical diagnostic tool "
        "and must not be used to make clinical decisions."
    )

    supported_types = sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS)
    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=supported_types,
        help="Supported formats: PNG, JPEG, TIFF, BMP, and WebP.",
    )
    if uploaded_file is None:
        if not CHECKPOINT_PATH.is_file():
            st.warning(
                f"Model checkpoint not found at `{CHECKPOINT_PATH}`. "
                "Train and save a compatible model before running predictions. "
                "The app will not start training automatically."
            )
        return

    try:
        with Image.open(uploaded_file) as source:
            image = source.copy()
            image.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        st.error(f"That file could not be decoded as an image: {exc}")
        return

    if uploaded_file.name.rsplit(".", 1)[-1].lower() not in supported_types:
        st.error("Unsupported image format. Please upload PNG, JPEG, TIFF, BMP, or WebP.")
        return

    st.subheader("Uploaded MRI")
    st.image(image, caption=uploaded_file.name, width="stretch")
    if not st.button("Generate segmentation", type="primary"):
        return

    if not CHECKPOINT_PATH.is_file():
        st.error(
            f"Cannot run inference because the checkpoint is missing: `{CHECKPOINT_PATH}`. "
            "Train the model separately, then restart or refresh this page."
        )
        return

    device = _device()
    try:
        model, checkpoint = _cached_model(str(CHECKPOINT_PATH), str(device))
        with st.spinner("Generating segmentation..."):
            mask, overlay = predict_image(image, model, checkpoint)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not run prediction: {exc}")
        return

    mask_column, overlay_column = st.columns(2)
    with mask_column:
        st.subheader("Predicted mask")
        st.image(mask, caption="Binary mask", width="stretch")
    with overlay_column:
        st.subheader("Overlay")
        st.image(overlay, caption="Red indicates predicted foreground", width="stretch")

    visualization = compose_visualization(image, mask, overlay)
    st.subheader("Side-by-side visualization")
    st.image(visualization, width="stretch")

    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    st.download_button(
        "Download predicted mask (PNG)",
        data=buffer.getvalue(),
        file_name=f"{Path(uploaded_file.name).stem}_pred_mask.png",
        mime="image/png",
    )
    visualization_buffer = BytesIO()
    visualization.save(visualization_buffer, format="PNG")
    st.download_button(
        "Download side-by-side visualization (PNG)",
        data=visualization_buffer.getvalue(),
        file_name=f"{Path(uploaded_file.name).stem}_visualization.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
