# MRI Segmentation Demo

An educational binary segmentation prototype built with PyTorch, U-Net, and
Streamlit. The app accepts an MRI image and displays the predicted mask and an
overlay. It is not a medical diagnostic tool.

## Run locally

From this project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.
Upload a PNG, JPEG, TIFF, BMP, or WebP image and click **Generate
segmentation**. The app uses `models/best_model.pth` and offers PNG downloads
for the mask and visualization.

## Model

The included checkpoint was trained for 15 epochs on 535 deduplicated image-mask
pairs from 19 patient groups. It uses RGB inputs resized to 128 × 128 and a
U-Net with 8 base channels. Patient groups were kept separate across the
training, validation, and test splits.

On the held-out test split (68 slices from 3 patient groups), macro Dice was
0.8456 across all slices and 0.7173 on the 23 slices with non-empty masks.
The all-slice score includes empty masks. These results are an initial project
evaluation and do not establish clinical reliability. Details are in
`outputs/evaluation_results.csv`; the epoch log is in
`outputs/training_history.csv`.

## Retrain or evaluate

The training dataset is intentionally excluded from Git. To retrain, place
paired files in separate image and mask directories, then run:

```powershell
.\.venv\Scripts\python.exe -m src.train --images data\training_all\images --masks data\training_all\masks --epochs 15 --batch-size 8 --image-size 128 128 --base-channels 8 --image-mode rgb --device cpu
```

Evaluate the held-out patient groups with:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --images data\training_all\images --masks data\training_all\masks --checkpoint models\best_model.pth --output outputs\evaluation_results.csv --device cpu
```

## Publish online

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for Streamlit Community Cloud setup. The
source MRI datasets and generated outputs are excluded from Git; the trained
checkpoint is included so the hosted app can run predictions.
