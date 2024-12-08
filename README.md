# **Emotion Recognition Using RAVDESS Audio Data**

## **Overview**
This project focuses on building a system to recognize human emotions from audio signals using the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). By processing speech audio files and applying machine learning models, the system classifies emotions such as happiness, sadness, anger, and more.

The project includes steps like feature extraction, dimensionality reduction, and model training to create a robust emotion classification system.

---

## **Dataset**
The dataset used is the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. This dataset includes:
- **1440 speech audio files** with emotions such as neutral, calm, happy, sad, angry, fearful, disgust, and surprised.
- Audio files are labeled based on the structured filename format.

### **Filename Structure**
Each file is named using the format:
`03-01-06-01-02-01-12.wav`

Where:
- **Modality**: `01` = full-AV, `02` = video-only, `03` = audio-only
- **Vocal Channel**: `01` = speech, `02` = song
- **Emotion**:  
  `01` = neutral,  
  `02` = calm,  
  `03` = happy,  
  `04` = sad,  
  `05` = angry,  
  `06` = fearful,  
  `07` = disgust,  
  `08` = surprised
- **Emotional Intensity**: `01` = normal, `02` = strong
- **Statement**: `01` = "Kids are talking by the door", `02` = "Dogs are sitting by the door"
- **Repetition**: `01` = 1st repetition, `02` = 2nd repetition
- **Actor**: Numbers `01`–`24` (odd = male, even = female)

Example:  
`03-01-06-01-02-01-12.wav`  
- **Audio-only**, **Speech**, **Fearful**, **Normal Intensity**, **Dogs Statement**, **1st Repetition**, **Actor 12 (Female)**.

---

## **Project Structure**
# Emotion Recognition Using Audio Features (YAMNet and Librosa)

## Overview
This project focuses on emotion recognition from audio files using both TensorFlow's YAMNet and manual feature extraction with Librosa. Advanced techniques like SMOTE, UMAP, and t-SNE are used for balancing, dimensionality reduction, and visualization. The dataset consists of `.wav` audio files organized into folders representing various emotions.

### Goals:
1. Extract audio embeddings using YAMNet.
2. Manually extract audio features using Librosa.
3. Balance the dataset using SMOTE.
4. Perform dimensionality reduction and visualization using t-SNE and UMAP.
5. Enable emotion classification using the extracted features.

---

## Dataset Structure
The dataset is organized as follows:
```
data/
├── Angry/
│   ├── sample1.wav
│   ├── sample2.wav
├── Happy/
│   ├── sample1.wav
│   ├── sample2.wav
├── Neutral/
│   ├── sample1.wav
│   ├── sample2.wav
└── ...
```
Each folder corresponds to an emotion category.

---

## Key Steps

### 1. Feature Extraction

#### Using YAMNet
Audio features are extracted using TensorFlow's YAMNet model:
- The model produces 1024-dimensional embeddings for each audio input.
- Features are averaged across all time frames to create a fixed-size representation for each audio file.

Code snippet:
```python
from tensorflow_hub import load
import librosa
import numpy as np

yamnet_model = load("https://tfhub.dev/google/yamnet/1")

def extract_yamnet_features(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform = waveform[:16000 * 10]  # Ensure <=10 seconds
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)
```

#### Using Librosa
Manually extracted features include MFCCs, deltas, spectral features, and prosodic features:
```python
import librosa
import numpy as np

def extract_librosa_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000)
    features = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.update({f"MFCC_{i+1}": np.mean(mfcc[i]) for i in range(n_mfcc)})

    # Delta and Delta-Delta (MFCC derivatives)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.update({f"Delta_MFCC_{i+1}": np.mean(delta_mfcc[i]) for i in range(n_mfcc)})
    features.update({f"Delta2_MFCC_{i+1}": np.mean(delta2_mfcc[i]) for i in range(n_mfcc)})

    # Spectral features
    features["Spectral_Centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["Spectral_Bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Prosodic features
    features["Zero_Crossing_Rate"] = np.mean(librosa.feature.zero_crossing_rate(y=y))

    return features
```

### 2. Balancing the Dataset
Using SMOTE to oversample minority classes, ensuring a balanced distribution of labels.

Code snippet:
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_smote))
```

### 3. Dimensionality Reduction and Visualization
**t-SNE** and **UMAP** are used to reduce the high-dimensional embeddings to 2D for visualization and exploratory analysis.

- **t-SNE:**
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_smote)
```

- **UMAP:**
```python
import umap.umap_ as umap

umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_smote)
```

- **Visualization:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
for label in Counter(y_smote).keys():
    plt.scatter(
        X_umap[y_smote == label, 0],
        X_umap[y_smote == label, 1],
        label=f"Class {label}"
    )
plt.title("UMAP Visualization of Balanced Dataset")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.show()
```

### 4. Dataset Saving
The balanced dataset with extracted features is saved for further analysis or modeling:
```python
balanced_data.to_csv("balanced_dataset.csv", index=False)
```

---

## Dependencies
- Python 3.11
- TensorFlow
- TensorFlow Hub
- NumPy
- Librosa
- Scikit-learn
- UMAP-learn
- Matplotlib
- Imbalanced-learn (SMOTE)

---

## Installation
1. Clone this repository.
2. Set up a Conda environment:
   ```bash
   conda create -n emotion python=3.11
   conda activate emotion
   ```
3. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-hub numpy librosa scikit-learn umap-learn matplotlib imbalanced-learn
   ```

---

## Usage
1. Organize your dataset into folders based on emotion labels.
2. Run the feature extraction script to generate embeddings using YAMNet or Librosa.
3. Balance the dataset using SMOTE.
4. Visualize the data with t-SNE or UMAP.
5. Train machine learning models on the balanced dataset for emotion classification.

---

## Output Files
- `yamnet_extracted_features.csv`: Contains extracted YAMNet features with labels.
- `librosa_extracted_features.csv`: Contains manually extracted features using Librosa with labels.
- `balanced_dataset.csv`: Balanced dataset ready for modeling.

---

## Future Improvements
- Experiment with additional feature extraction techniques (e.g., spectrogram-based CNNs).
- Explore deep learning-based classification.
- Integrate a web interface for real-time emotion recognition.

---

