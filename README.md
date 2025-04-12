# Offensive Language Classification

## Project Overview
This project develops machine learning models to detect toxic content in online feedback, focusing on binary classification (`toxic` vs. `non-toxic`). The task mirrors real-world content moderation challenges, requiring robust handling of multilingual text. Two notebooks are implemented:
- **Model 1**: Logistic Regression (baseline) and GRU (sequential model).
- **Model 2**: BERT (transformer-based model for multilingual support).

The goal is to predict the `toxic` label accurately, leveraging fine-grained labels (`abusive`, `vulgar`, etc.) during training to enhance model understanding.

## Dataset Description
The dataset consists of three files:
- **train.csv**: Labeled training data with columns:
  - `id`: Unique comment identifier.
  - `feedback_text`: Comment text (mostly English).
  - `toxic`, `abusive`, `vulgar`, `menace`, `offense`, `bigotry`: Binary labels (0 = absent, 1 = present).
- **validation.csv**: Multilingual validation data with `id`, `feedback_text`, `lang`, and `toxic` columns.
- **test.csv**: Unlabeled multilingual test data with `id`, `content`, and `lang` columns.

Multiple labels can be active per comment, but evaluation focuses solely on the `toxic` label.

## Model Implementation Details
### Model 1: Logistic Regression + GRU (`model1_implementation.ipynb`)
- **Logistic Regression**:
  - **Preprocessing**: TF-IDF vectorization (5000 features) after lowercasing, stop word removal, and lemmatization.
  - **Training**: Trained on `toxic` label with `max_iter=1000`.
  - **Tuning**: Grid Search over `C=[0.1, 1, 10]`, best `C=10`.
  - **Output**: Saved as `lr_model.pkl`.
- **GRU**:
  - **Preprocessing**: Tokenized and padded sequences (max length 100, vocab size 5000).
  - **Architecture**: Bidirectional GRU with 64 and 32 units, embedding layer (128 dims), dropout (0.3), and sigmoid output.
  - **Training**: 5 epochs, batch size 32, Adam optimizer, binary cross-entropy loss.
  - **Output**: Saved as `gru_model.h5`.
- **Features**:
  - EDA: Label distribution, text length, word frequency visualizations.
  - Evaluation: Accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, ROC curve.
  - Test predictions: Saved as `submission_lr.csv` and `submission_gru.csv`.

### Model 2: BERT (`model2_implementation.ipynb`)
- **Preprocessing**: Used `bert-base-multilingual-cased` tokenizer (max length 128, truncation, padding).
- **Architecture**: Fine-tuned `bert-base-multilingual-cased` for binary classification (`num_labels=2`).
- **Training**:
  - Initial: 3 epochs, batch size 16, warmup steps 500, weight decay 0.01.
  - Tuned: 4 epochs, batch size 8.
- **Features**:
  - EDA: Same as Model 1 for consistency.
  - Evaluation: Accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, ROC curve.
  - Test predictions: Saved as `submission_bert.csv`.
  - Output: Saved as `bert_model` directory.
- **Notes**: Disabled Weights & Biases logging (`report_to="none"`) for local execution.

## Steps to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nur35982/Offensive_Language_Classification.git
