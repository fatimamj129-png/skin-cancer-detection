# Streamlit Cloud deployment
1. Create a GitHub repo and upload ALL files in this folder to its root (app.py, requirements.txt, class_names.json, model_info.json, model file(s)).
2. GitHub rejects files > 100 MB. If skin_cancer_resnet50.keras is larger, either use Git LFS for it,
   or delete it and keep only skin_cancer_resnet50.tflite (the app falls back to TFLite automatically).
3. Go to https://share.streamlit.io -> New app -> pick the repo -> main file: app.py -> Deploy.
4. In Advanced settings choose a Python version supported by this TensorFlow version (3.11 or 3.12 usually works).
