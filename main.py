from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from database import init_db, save_checked_email, save_checked_url, get_email_history, get_url_history

app = FastAPI(title="Hybrid Phishing Detection API with Secure Database", version="1.0")

# تهيئة قاعدة البيانات عند بدء التشغيل
init_db()

# ==========================================
# تحميل نموذج الروابط فقط عند الإقلاع (خفيف ولا يستهلك رام)
# ==========================================
URL_MODEL_PATH = "my_phishing_url_model.pkl"
url_model = joblib.load(URL_MODEL_PATH)
print("✅ تم تحميل نموذج الروابط بنجاح.")

# ==========================================
# التحميل الكسول لنموذج BERT (يتحمل فقط عند الحاجة لتفريغ الذاكرة)
# ==========================================
BERT_MODEL_DIR = "shomoukh/my_phishing_bert_model"
tokenizer = None
bert_model = None

def get_bert_model():
    global tokenizer, bert_model
    if bert_model is None or tokenizer is None:
        print("⏳ جاري تحميل نموذج BERT لأول طلب فحص...")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_DIR,
            torch_dtype=torch.float16
        )
        bert_model.eval()
        print("✅ تم تحميل نموذج BERT بنجاح.")
    return tokenizer, bert_model

class ScanRequest(BaseModel):
    user_id: int
    email_text: str
    url_to_check: str
    sender_address: str = None

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

@app.get("/")
def home():
    return {"message": "Welcome to the Hybrid Phishing Detection API with Secure Database!"}

@app.post("/scan-complete")
def scan_complete(data: ScanRequest):
    results = {}
    
    # 1. فحص الرابط وحفظه تلقائياً في قاعدة البيانات
    try:
        url_features = extract_url_features(data.url_to_check)
        url_pred = url_model.predict(url_features)[0]
        url_prob = url_model.predict_proba(url_features)[0].max()
        
        is_phish_url = bool(url_pred)
        conf_url = float(url_prob)
        
        save_checked_url(
            user_id=data.user_id,
            url_content=data.url_to_check,
            is_phishing=is_phish_url,
            confidence=conf_url
        )
        
        results["url_analysis"] = {
            "url": data.url_to_check,
            "is_phishing": is_phish_url,
            "confidence": conf_url
        }
    except Exception as e:
        results["url_analysis"] = {"error": str(e)}

    # 2. فحص الإيميل بنموذج الـ BERT (يتم تحميله عند الطلب حصرياً لتوفير الرام)
    try:
        cur_tokenizer, cur_bert_model = get_bert_model()
        
        inputs = cur_tokenizer(data.email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = cur_bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()
            
        is_phish_email = bool(pred_label)
        conf_email = float(confidence)
        
        save_checked_email(
            user_id=data.user_id,
            email_content=data.email_text,
            is_phishing=is_phish_email,
            confidence=conf_email,
            sender_address=data.sender_address,
            source_note="BERT Email Model"
        )
        
        results["email_analysis"] = {
            "is_phishing": is_phish_email,
            "confidence": conf_email
        }
    except Exception as e:
        results["email_analysis"] = {"error": str(e)}

    return {"status": "success", "results": results}

@app.get("/history/{user_id}")
def get_history(user_id: int):
    emails_history = get_email_history(user_id)
    urls_history = get_url_history(user_id)
    return {
        "user_id": user_id,
        "emails_history": emails_history,
        "urls_history": urls_history
    }