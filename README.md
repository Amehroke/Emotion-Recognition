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
