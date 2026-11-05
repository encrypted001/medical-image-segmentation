# Publish the MRI segmentation demo

This app can be deployed with Streamlit Community Cloud. The repository needs
`src/app.py`, `requirements.txt`, and `models/best_model.pth`. The `.gitignore`
keeps the source MRI datasets, prepared training copy, and generated results out
of Git while allowing the demo checkpoint to be included.

## Deploy

1. Create a GitHub repository for this project. Keep it private if you prefer
   not to publish the source code. Do not add the ignored dataset folders.
2. Push the project files, including `models/best_model.pth`, to GitHub.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and choose **Create app**.
4. Select the repository and branch, and set the app file to `src/app.py`.
5. In **Advanced settings**, select Python 3.14 to match the environment used
   for this checkpoint, then deploy.

The app accepts uploaded images for prediction and does not need the training
data on the hosting service. It is an educational research prototype, not a
medical diagnostic tool. Avoid uploading patient-identifiable scans to a public
demo.
