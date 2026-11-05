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

The included checkpoint uses RGB inputs resized to 128 x 128 and a U-Net with
8 base channels. Training used mild brightness/contrast changes, small rotations,
and horizontal flips. The local dataset had 472 paired slices from 19 patient
groups, split by patient into 326 training, 78 validation, and 68 test slices.

On the grouped test split (68 slices from 3 patient groups), macro Dice was
0.8607 and macro IoU was 0.8263. On the 23 slices with non-empty masks, mean
Dice was 0.7620 and mean IoU was 0.6604. Four of 45 empty-mask slices had a
non-empty prediction. The all-slice macro scores include empty masks and should
be read alongside the positive-slice scores. This is an exploratory result from
a small patient cohort; multiple experiments used the same split, so an external
cohort is needed for an independent accuracy claim. These metrics do not establish
clinical reliability.

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
