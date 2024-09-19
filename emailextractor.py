import imaplib
import email
from email.header import decode_header
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")
imap_server = os.getenv("IMAP_SERVER")

mail = imaplib.IMAP4_SSL(imap_server)
mail.login(username, password)
mail.select("inbox")

today = datetime.now().strftime("%d-%b-%Y")

status, messages = mail.search(None, f'(SINCE "{today}")')
email_ids = messages[0].split()

def fetch_emails(email_ids):
    emails_content = []
    for email_id in email_ids:
        _, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else 'utf-8', errors='ignore')
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if "attachment" not in content_disposition:
                            try:
                                body = part.get_payload(decode=True)
                                if body:
                                    try:
                                        body = body.decode('utf-8')
                                    except UnicodeDecodeError:
                                        body = body.decode('latin-1', errors='ignore')
                                    emails_content.append({"subject": subject, "body": body})
                            except Exception as e:
                                print(f"Erro ao decodificar o corpo do e-mail: {e}")
                else:
                    body = msg.get_payload(decode=True)
                    if body:
                        try:
                            body = body.decode('utf-8')
                        except UnicodeDecodeError:
                            body = body.decode('latin-1', errors='ignore')
                        emails_content.append({"subject": subject, "body": body})
    return emails_content

emails_content = fetch_emails(email_ids)
print(emails_content)

mail.logout()

from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    return predicted_class

for email in emails_content:
    subject = email["subject"]
    body = email["body"]
    print(f"Assunto: {subject}")
    print(f"Classificação: {classify_email(body)}")
    print("-------------------------------------------------")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id

def summarize_email(text):
    input_ids = gpt2_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
    attention_mask = (input_ids != gpt2_model.config.pad_token_id).to(dtype=torch.long)

    summary_ids = gpt2_model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2
    )
    
    summary = gpt2_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

for email in emails_content:
    subject = email["subject"]
    body = email["body"]
    print(f"Assunto: {subject}")
    print(f"Classificação: {classify_email(body)}")
    print(f"Resumo: {summarize_email(body)}")
    print("-------------------------------------------------")

import pandas as pd

def create_report(emails):
    data = []
    for email in emails:
        subject = email["subject"]
        body = email["body"]
        classification = classify_email(body)
        summary = summarize_email(body)
        data.append({"Assunto": subject, "Classificação": classification, "Resumo": summary})
    
    df = pd.DataFrame(data)
    df.to_csv("email_report.csv", index=False)
    print("Relatório salvo como 'email_report.csv'")

create_report(emails_content)
