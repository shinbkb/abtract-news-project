import re, unicodedata, hashlib

def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text)).lower()).strip()

def fingerprint(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()

def tokenize(text):
    text=unicodedata.normalize('NFC',str(text).lower())
    return re.findall(r'[^\W_]+(?:_[^\W_]+)*',text,flags=re.UNICODE)
