import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from dotenv import load_dotenv
import torch
import os

load_dotenv()

username = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")
imap_server = os.getenv("IMAP_SERVER")

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def summarize_email_bart(text):
    try:
        inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        
        summary_ids = bart_model.generate(
            inputs['input_ids'], 
            max_length=150,
            min_length=30, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        if len(summary) > 500:
            summary = summary[:500]
        
        return summary
    except Exception as e:
        print(f"Erro ao resumir o e-mail com BART: {e}")
        return "Erro ao resumir o e-mail."

def fetch_emails(mail):
    emails_content = []
    try:
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%d-%b-%Y")
        
        status, messages = mail.search(None, f'(SINCE "{two_days_ago}")')
        email_ids = messages[0].split()

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
                                body = part.get_payload(decode=True)
                                if body:
                                    try:
                                        body = body.decode('utf-8', errors='ignore')
                                        emails_content.append({"subject": subject, "body": body})
                                    except Exception as e:
                                        print(f"Erro ao decodificar o corpo do e-mail: {e}")
                    else:
                        body = msg.get_payload(decode=True)
                        if body:
                            try:
                                body = body.decode('utf-8', errors='ignore')
                                emails_content.append({"subject": subject, "body": body})
                            except Exception as e:
                                print(f"Erro ao decodificar o corpo do e-mail: {e}")
        return emails_content
    except Exception as e:
        print(f"Erro ao buscar e-mails: {e}")
        return emails_content

def create_report(emails):
    data = []
    for email in emails:
        subject = email.get("subject", "Sem Assunto")
        body = email.get("body", "")
        
        summary = summarize_email_bart(body)
        
        data.append({"Assunto": subject, "Resumo": summary})
    
    df = pd.DataFrame(data)
    df.to_csv("email_report.csv", index=False)
    print("Relatório salvo como 'email_report.csv'")

def main():
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(username, password)
        mail.select("inbox")
        
        emails_content = fetch_emails(mail)
        mail.logout()

        if emails_content:
            create_report(emails_content)
        else:
            print("Nenhum e-mail encontrado para o período.")
    except Exception as e:
        print(f"Erro geral: {e}")

if __name__ == "__main__":
    main()
