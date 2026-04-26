# Sugarcane Leaf Disease Classification

End-to-end TensorFlow project for training, evaluating, explaining, serving, and exporting a sugarcane leaf disease classifier.

The training script compares multiple approaches and selects the best model using validation accuracy and macro F1:

- Custom CNN
- MobileNetV2 transfer learning
- EfficientNetB0 transfer learning
- ResNet50 transfer learning
- Compact Vision Transformer

## Dataset Layout

Place your images in this folder structure:

```text
dataset/
  train/
    Healthy/
    Rust/
    RedRot/
    LeafScald/
    MosaicDisease/
    YellowLeafDisease/
    EyeSpot/
    BrownSpot/
    OtherUnknownDisease/
  val/
    Healthy/
    Rust/
    ...
  test/
    Healthy/
    Rust/
    ...
```

Class folders can use spaces, dashes, or underscores. Keep the same class folders across `train`, `val`, and `test`.

If you do not have data yet, create a tiny synthetic smoke-test dataset:

```bash
python scripts/simulate_dataset.py --output dataset --images-per-class 30
```

This synthetic data is only for checking that the pipeline runs. It is not a real agricultural model.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow will use GPU automatically when a compatible CUDA setup is available. Mixed precision is enabled automatically on GPU unless disabled.

## Train And Select Best Model

```bash
python -m src.train --data-dir dataset --models custom_cnn mobilenetv2 efficientnetb0 resnet50 vit --epochs 25
```

Outputs are saved under `artifacts/`:

- `artifacts/best_model.keras`
- `artifacts/best_model_info.json`
- `artifacts/class_names.json`
- per-model checkpoints and histories

For a faster first run:

```bash
python -m src.train --data-dir dataset --models mobilenetv2 efficientnetb0 --epochs 10 --img-size 224
```

## Evaluate

```bash
python -m src.evaluate --data-dir dataset --model artifacts/best_model.keras
```

Generated files:

- `artifacts/evaluation/classification_report.csv`
- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/confusion_matrix.png`
- `artifacts/evaluation/roc_curve.png`

## Predict

```bash
python -m src.predict --image path\to\leaf.jpg --model artifacts/best_model.keras
```

Programmatic use:

```python
from src.predict import predict_leaf

result = predict_leaf("image.jpg")
print(result["final_prediction"])
print(result["scores"])
```

The predictor supports an unknown result when the best confidence is below the threshold.

## Grad-CAM Explainability

```bash
python -m src.gradcam --image path\to\leaf.jpg --model artifacts/best_model.keras
```

This writes a heatmap overlay to `artifacts/gradcam/`.

## API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Prediction endpoint:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

## Streamlit UI

```bash
streamlit run src/app_streamlit.py
```

## Export Mobile Model

Export a lightweight MobileNetV2 model to TensorFlow Lite:

```bash
python -m src.export_tflite --model artifacts/best_model.keras
```

If the best model is not MobileNetV2, train a MobileNetV2 model first or export the best compatible Keras model:

```bash
python -m src.train --data-dir dataset --models mobilenetv2 --epochs 15
python -m src.export_tflite --model artifacts/mobilenetv2/best.keras
```

## Real-World Robustness Notes

This project includes rotation, flipping, brightness, contrast, zoom, blur, and light noise augmentations. For field deployment, improve reliability by:

- collecting images from multiple phones, farms, times of day, and growth stages
- keeping a strong `OtherUnknownDisease` class
- adding nutrient-deficiency examples if you want disease-vs-deficiency detection
- setting `--unknown-threshold` higher for cautious predictions
- validating on farms not used in training
