import pandas as pd
import numpy as np
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import streamlit as st
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Data Preprocessing ─────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['text', 'sentiment'], inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])  # negative=0, neutral=1, positive=2
    return df, le

# ── Dataset Class ──────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Model Training ─────────────────────────────────────────────────────────────
def train_model(train_loader, val_loader, num_labels):
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch['label'].numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_sentiment.pt')

    return model

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, le):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())

    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('Confusion Matrix - BERT Sentiment Analysis')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# ── Streamlit Dashboard ────────────────────────────────────────────────────────
def run_dashboard():
    st.set_page_config(page_title='NLP Sentiment Analyzer', layout='wide')
    st.title('NLP Sentiment Analysis Dashboard')
    st.markdown('**Powered by BERT + HuggingFace Transformers**')

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.load_state_dict(torch.load('best_bert_sentiment.pt', map_location=DEVICE))
    model.eval()

    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Single Text Prediction')
        user_text = st.text_area('Enter review or tweet:', height=150)
        if st.button('Analyze Sentiment'):
            encoding = tokenizer.encode_plus(
                clean_text(user_text), max_length=MAX_LEN, truncation=True,
                padding='max_length', return_tensors='pt'
            )
            with torch.no_grad():
                outputs = model(**encoding)
                pred = torch.argmax(outputs.logits, dim=1).item()
            st.success(f'Predicted Sentiment: **{label_map[pred]}**')

    with col2:
        st.subheader('Batch CSV Analysis')
        uploaded = st.file_uploader('Upload CSV with text column', type='csv')
        if uploaded:
            df = pd.read_csv(uploaded)
            df['clean'] = df['text'].apply(clean_text)
            preds = []
            for txt in df['clean']:
                enc = tokenizer.encode_plus(txt, max_length=MAX_LEN, truncation=True,
                                            padding='max_length', return_tensors='pt')
                with torch.no_grad():
                    out = model(**enc)
                    preds.append(label_map[torch.argmax(out.logits, dim=1).item()])
            df['sentiment_prediction'] = preds
            st.dataframe(df[['text', 'sentiment_prediction']].head(50))

            st.subheader('Sentiment Distribution')
            fig, ax = plt.subplots()
            df['sentiment_prediction'].value_counts().plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)

            st.subheader('Word Cloud')
            wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean']))
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)

# ── Main Pipeline ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        run_dashboard()
    else:
        # Training pipeline
        df, le = load_and_preprocess('data/sentiment_data.csv')
        print(f'Dataset shape: {df.shape}')
        print(f'Sentiment distribution:\n{df["sentiment"].value_counts()}')

        X_train, X_temp, y_train, y_temp = train_test_split(
            df['clean_text'].values, df['label'].values, test_size=0.3, random_state=42, stratify=df['label']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        train_ds = SentimentDataset(X_train, y_train, tokenizer, MAX_LEN)
        val_ds = SentimentDataset(X_val, y_val, tokenizer, MAX_LEN)
        test_ds = SentimentDataset(X_test, y_test, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = train_model(train_loader, val_loader, num_labels=len(le.classes_))
        evaluate_model(model, test_loader, le)

        joblib.dump(le, 'label_encoder.pkl')
        print('Training complete. Model saved as best_bert_sentiment.pt')
