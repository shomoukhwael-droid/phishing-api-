from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI(title="Hybrid Phishing Detection API", version="1.0")

# ==========================================
# 1. تحميل نموذج الروابط (Random Forest)
# ==========================================
URL_MODEL_PATH = "my_phishing_url_model.pkl"
url_model = joblib.load(URL_MODEL_PATH)
print("✅ تم تحميل نموذج الروابط بنجاح.")

# ==========================================
# 2. تحميل نموذج الإيميلات (BERT)
# ==========================================
# الاسم مطابق تماماً لاسم المجلد عندك في مجلد المشروع
BERT_MODEL_DIR = "my_phishing_bert_model" # أو اسم المجلد لديك

try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.eval()
    print("✅ تم تحميل نموذج BERT بنجاح.")
except Exception as e:
    print(f"⚠️ خطأ في تحميل نموذج BERT: {e}")

# هيكل البيانات المستقبلة من تطبيق Flutter
class ScanRequest(BaseModel):
    email_text: str
    url_to_check: str

# دالة استخراج خصائص الرابط
def extract_url_features(url):
    features = {}
    features['length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['has_ip'] = 1 if any(char.isdigit() for char in url.split('/')[0]) else 0
    features['is_https'] = 1 if url.startswith('https') else 0
    
    keywords = ['login', 'verify', 'update', 'secure', 'account', 'banking', 'signin', 'support', 'ebayisapi']
    features['has_suspicious_keyword'] = 1 if any(kw in url.lower() for kw in keywords) else 0
    
    return [list(features.values())]

# ==========================================
# 3. نقاط النهاية للـ API
# ==========================================
@app.get("/")
def home():
    return {"message": "Welcome to the Hybrid Phishing Detection API!"}

@app.post("/scan-complete")
def scan_complete(data: ScanRequest):
    results = {}
    
    # فحص الرابط
    try:
        url_features = extract_url_features(data.url_to_check)
        url_pred = url_model.predict(url_features)[0]
        url_prob = url_model.predict_proba(url_features)[0].max()
        
        results["url_analysis"] = {
            "url": data.url_to_check,
            "is_phishing": bool(url_pred),
            "confidence": float(url_prob)
        }
    except Exception as e:
        results["url_analysis"] = {"error": str(e)}

    # فحص الإيميل بـ BERT
    try:
        inputs = tokenizer(data.email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()
            
        results["email_analysis"] = {
            "is_phishing": bool(pred_label),
            "confidence": float(confidence)
        }
    except Exception as e:
        results["email_analysis"] = {"error": str(e)}

    return results