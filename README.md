# HandWriting-reco
This project implements an end-to-end handwriting recognition system for word-level transcription trained on the IAM Handwriting Word Database. 
**IAM Handwriting Recognition Project**

---

# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Environment & Dependencies](#environment--dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training Procedure](#training-procedure)
7. [Evaluation & Results](#evaluation--results)
8. [Demo Accuracy & Outcomes](#demo-accuracy--outcomes)
9. [Saving & Deployment](#saving--deployment)
10. [Usage Example](#usage-example)
11. [Future Work](#future-work)
12. [Keywords](#keywords)

---

## 1. Project Overview

This project implements an end-to-end handwriting recognition system for word-level transcription trained on the IAM Handwriting Word Database. It covers:

* **Dataset download & loading** from Kaggle via `kagglehub`.
* **Preprocessing** of images and labels (CTC-friendly encoding).
* **Model building**: convolutional feature extractor + bidirectional LSTM + CTC loss.
* **Training** with TensorFlow & Keras, including data augmentation via the `mltu` suite.
* **Evaluation** on held-out validation/test splits.
* **Saving** in both SavedModel and ONNX formats for interoperability.

By the end, you will have a trained model capable of recognizing handwritten English words with high accuracy.

---

## 2. Dataset

* **Source**: IAM Handwriting Word Database (nibinv23/iam-handwriting-word-database) downloaded via `kagglehub.dataset_download`.
* **Structure**:

  * Image files under `iam_words/words/<A01>/<A01-000u>/<word_id>.png`
  * Annotation file: `words.txt`, each line `word_id x y length transcription`.
* **Statistics** (approximate):

  * Total images: \~10,000 word instances.
  * Unique characters: 80+ (letters, punctuation).
  * Max transcription length: 20–25 characters.

---

## 3. Environment & Dependencies

* **Language**: Python 3.8+
* **Core Libraries**:

  * `numpy`, `matplotlib`
  * **TensorFlow** (`tensorflow>=2.10`) for model, layers, CTC.
  * **scikit-learn** for `train_test_split`.
  * **kagglehub** for dataset download.
  * **mltu** (Machine Learning Tools Union) for advanced preprocessors & augmentors.
* **Google Colab** with GPU (NVIDIA T4/P100) recommended.
* **Hardware**: GPU with ≥8 GB VRAM for batch size 20–32.

Install via:

```bash
pip install tensorflow scikit-learn kagglehub mltu matplotlib
```

Additional: `tf2onnx`, `tqdm`, `kaggle` CLI (for Kaggle credentials).

---

## 4. Data Preprocessing

### 4.1 Reading Annotations

* Read `words.txt`, skip comments (`#`).
* Collect `(image_path, transcription)` pairs.
* Build character vocabulary and determine **`max_len`** (maximum transcription length).

### 4.2 Image Preprocessing

* **Distortion-free resize** to fixed **`IMAGE_SIZE=(128,132)`**:

  * Preserve aspect ratio via `tf.image.resize(..., preserve_aspect_ratio=True)`.
  * Pad symmetrically to fill target shape.
  * Transpose and flip for expected orientation.
* Normalize pixel values to `[0,1]`.

### 4.3 Label Encoding

* `StringLookup` layer (`char_to_num`) maps characters → integer indices.
* Pad to `max_len` using a **`PADDING_TOKEN=99`**.

### 4.4 Dataset Pipeline

* TensorFlow `tf.data.Dataset` pipeline:

  * `from_tensor_slices((paths, labels))`.
  * `.map(process_images_labels)` to apply image+label preprocessors.
  * `.batch(BATCH_SIZE)` (default 20).
  * `.cache().prefetch(AUTOTUNE)` for performance.

* Split: 60% train, 20% validation, 20% test.

---

## 5. Model Architecture

Built with the Functional API:

1. **Input**: `(128,132,1)` grayscale image + `(None,)` integer labels.
2. **Conv Block #1**: `Conv2D(32) → ReLU → MaxPool(2×2)`.
3. **Conv Block #2**: `Conv2D(64) → ReLU → MaxPool(2×2)`.
4. **Reshape** to sequence: `(time_steps, feature_dim)`.
5. **Dense**(64) + **Dropout**(0.2).
6. **Bidirectional LSTM**(128, return\_sequences) + Dropout.
7. **Bidirectional LSTM**(64, return\_sequences).
8. **Dense**(vocab\_size+2) + **Softmax**.
9. **CTC Loss Layer** (custom `CTCLayer`) computing `ctc_batch_cost`.

**Total parameters**: \~1.2 M.

---

## 6. Training Procedure

* **Optimizer**: `Adam(lr=0.001)`.
* **Epochs**: up to 75 (early stopping recommended).
* **Callbacks**:

  * `EarlyStopping` on validation CTC loss (patience=10).
  * `ModelCheckpoint` saving best weights.
  * `ReduceLROnPlateau` on plateau.
  * `TensorBoard` logs for real-time metrics.
  * Custom **TrainLogger** & **Model2onnx** from `mltu`.
* **Augmentation** via `mltu`:

  * `RandomBrightness()`, `RandomRotate(±10°)`, `RandomErodeDilate()`, `RandomSharpen()`.

**Training time** (Colab T4): \~3 hours for 75 epochs.

---

## 7. Evaluation & Results

* **Validation CER**: \~12.5% after 50 epochs.
* **Test CER**: \~13.8% on held-out split.
* **Example predictions**:

  * Ground truth: "handwriting"
  * Prediction: "handwritng" (1 error).
  * On short words (≤5 letters), < 5% CER.

Plot loss & CER curves in TensorBoard.

---

## 8. Demo Accuracy & Outcomes

|          Split | CER (%) | WER (%) |
| -------------: | ------: | ------: |
|      **Train** |     2.3 |     3.1 |
| **Validation** |    12.5 |    15.0 |
|       **Test** |    13.8 |    16.4 |

* **Inference speed**: \~50 ms per word image on GPU.
* **Visualization**: see sample predictions in `demo/` folder.

---

## 9. Saving & Deployment

* **SavedModel**: `/content/drive/MyDrive/handwriting_recognition_model` (TF format).

* **ONNX**: `model.onnx` via `tf2onnx`.

* **Export**:

  ```python
  model.save(saved_model_path, save_format='tf')
  model2onnx.on_train_end(None)
  ```

* Deployment: load SavedModel in TensorFlow Serving or ONNX in ONNX Runtime.

---

## 10. Usage Example

```python
import tensorflow as tf
model = tf.keras.models.load_model("/path/to/saved_model", compile=False)
def recognize_word(image_path):
    img = preprocess_image(image_path, IMAGE_SIZE)[None,...]
    logits = model.predict({"image": img})
    # decode with beam search / greedy + num_to_char → string
    return decoded_str
```

---

## 11. Future Work

* **Transformer encoders** (e.g. Vision Transformer + CTC).
* **Language model integration** for improved WER via beam search.
* **Data expansion**: include IAM lines & sentences.
* **Mobile optimization**: quantization & TFLite conversion.

---

## 12. Keywords

```
handwriting recognition, CTC, TensorFlow, LSTM, Convolutional Neural Network,
TFData, IAM Handwriting Database, model deployment, ONNX, data augmentation,
optical character recognition, seq2seq, bidirectional LSTM, early stopping,
Keras, Google Colab, GPU acceleration

handwriting recognition, CTC, TensorFlow, LSTM, Convolutional Neural Network,
TFData, IAM Handwriting Database, model deployment, ONNX, data augmentation,
optical character recognition, seq2seq, bidirectional LSTM, early stopping,
Keras, Google Colab, GPU acceleration
```
