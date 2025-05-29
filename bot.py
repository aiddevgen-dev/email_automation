import os
import imaplib
import email
import smtplib
import pandas as pd
import openai
import streamlit as st
from datetime import datetime, timedelta
import re
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import html
import time
import io
import sys
import traceback

# Streamlit configuration MUST BE FIRST
st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Drive & OAuth imports with error handling
GOOGLE_DRIVE_AVAILABLE = False
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError as e:
    print(f"Google Drive libraries not available: {e}")

# Configuration
IMAP_SERVER, SMTP_SERVER = 'imap.gmail.com', 'smtp.gmail.com'
CSV_FILE, LOG_FILE = 'emails.csv', 'app.log'
CUSTOM_PROMPT_FILE, DRIVE_CREDS_FILE, DRIVE_TOKEN_FILE = (
    'custom_prompt.json',
    'drive_credentials.json',
    'drive_token.json'
)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Logging setup
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    )
except Exception as e:
    print(f"Logging setup error: {e}")

def log(msg, level="info"):
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        getattr(logging, level)(msg)
    except Exception as e:
        print(f"Logging error: {e} - Message: {msg}")

def load_custom_css():
    try:
        st.markdown("""
        <style>
        .main { padding-top: 2rem; }
        .metric-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 1rem; border-radius: 8px; text-align: center; 
        }
        .custom-prompt-box { 
            background: #f0f8ff; border: 2px solid #4a90e2; 
            border-radius: 8px; padding: 1rem; margin: 1rem 0; 
        }
        .stExpander { 
            border: 1px solid #e0e0e0; border-radius: 8px; margin: 0.5rem 0; 
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        log(f"CSS loading error: {e}", "error")

# Gmail Functions
def login_gmail(user, password, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not user or not password:
                raise Exception("Gmail credentials are required")
            
            mail = imaplib.IMAP4_SSL(IMAP_SERVER, 993)
            mail.sock.settimeout(60)
            mail.login(user, password)
            log(f"Successfully logged in to Gmail for {user}")
            return mail
        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = f"Gmail login failed after {max_retries} attempts: {str(e)}"
                log(error_msg, "error")
                raise Exception(error_msg)
            log(f"Login attempt {attempt + 1} failed, retrying: {str(e)}", "warning")
            time.sleep(2)

def clean_html_text(text):
    if not text: 
        return ""
    try:
        text = html.unescape(text)
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            for script in soup(["script", "style"]): 
                script.decompose()
            text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        log(f"HTML cleaning error: {e}", "warning")
        return str(text) if text else ""

def get_body(msg):
    try:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                try:
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
                    elif content_type == 'text/html' and not body:
                        body = clean_html_text(part.get_payload(decode=True).decode(errors='ignore'))
                except Exception as e:
                    log(f"Error processing email part: {e}", "warning")
                    continue
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')
                if msg.get_content_type() == 'text/html':
                    body = clean_html_text(body)
        return clean_html_text(body)
    except Exception as e:
        log(f"Error extracting email body: {str(e)}", "error")
        return ""

def fetch_emails(mail, folder='inbox', limit=20, progress_callback=None):
    try:
        folder_name = folder if folder != 'sent' else '"[Gmail]/Sent Mail"'
        mail.select(folder_name)
        
        _, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split() if messages[0] else []
        if not email_ids: 
            log(f"No emails found in {folder}", "warning")
            return []
        
        email_ids = email_ids[-limit:]  # Get latest emails
        emails = []
        
        for i, eid in enumerate(email_ids):
            if progress_callback:
                progress = (i / len(email_ids)) * 100
                progress_callback(progress, f"Processing email {i+1}/{len(email_ids)}")
            
            try:
                mail.sock.settimeout(30)
                _, data = mail.fetch(eid, '(RFC822)')
                if not data or not data[0] or not data[0][1]:
                    continue
                
                msg = email.message_from_bytes(data[0][1])
                email_date = msg.get('Date', '')
                try:
                    from email.utils import parsedate_to_datetime
                    parsed_date = parsedate_to_datetime(email_date)
                except:
                    parsed_date = datetime.now()
                
                emails.append({
                    'id': eid.decode(),
                    'subject': str(msg.get('Subject', ''))[:200],
                    'from': str(msg.get('From', ''))[:100],
                    'to': str(msg.get('To', ''))[:100],
                    'date': email_date,
                    'parsed_date': parsed_date,
                    'body': get_body(msg)[:2000]
                })
            except Exception as e:
                log(f"Error processing email {eid}: {e}", "warning")
                continue
        
        log(f"Fetched {len(emails)} emails from {folder}")
        return emails
    except Exception as e:
        log(f"Error fetching emails: {str(e)}", "error")
        return []

def match_replies(inbox_emails, sent_emails):
    pairs = []
    try:
        for inbox_email in inbox_emails:
            try:
                customer_from = str(inbox_email.get('from', '')).lower()
                customer_email = re.search(r'<(.+?)>', customer_from)
                customer_email = customer_email.group(1) if customer_email else customer_from
                reply = ''
                
                for sent_email in sent_emails:
                    if customer_email in str(sent_email.get('to', '')).lower():
                        reply = sent_email.get('body', '')
                        break
                
                pairs.append({
                    'subject': inbox_email.get('subject', ''),
                    'from': inbox_email.get('from', ''),
                    'date': inbox_email.get('date', ''),
                    'parsed_date': inbox_email.get('parsed_date', datetime.now()),
                    'customer': str(inbox_email.get('body', '')).strip(),
                    'reply': str(reply).strip(),
                    'has_reply': bool(str(reply).strip())
                })
            except Exception as e:
                log(f"Error matching email: {e}", "warning")
                continue
        
        log(f"Matched {len(pairs)} email pairs")
        return pairs
    except Exception as e:
        log(f"Error matching replies: {str(e)}", "error")
        return []

def send_email(user, password, to, subject, body):
    try:
        if not user or not password or not to:
            return False, "Missing email credentials or recipient"
        
        msg = MIMEMultipart()
        msg['From'], msg['To'] = user, to
        msg['Subject'] = f"Re: {subject}" if not subject.lower().startswith('re:') else subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        with smtplib.SMTP(SMTP_SERVER, 587) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        
        log(f"Email sent successfully to {to}")
        return True, "Email sent successfully"
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        log(error_msg, "error")
        return False, error_msg

# Knowledge Base Functions
def extract_keywords(text):
    try:
        if not text: 
            return []
        patterns = [
            r'\b(refund|return|cancel|exchange)\b',
            r'\b(order|purchase|payment|billing)\b',
            r'\b(shipping|delivery|tracking)\b',
            r'\b(problem|issue|help|support)\b',
            r'\b(account|login|password)\b',
            r'\b(complaint|feedback|review)\b'
        ]
        keywords = []
        for pattern in patterns:
            keywords.extend(re.findall(pattern, str(text).lower(), re.IGNORECASE))
        return list(set(keywords))
    except Exception as e:
        log(f"Keyword extraction error: {e}", "warning")
        return []

def load_custom_prompt():
    try:
        if os.path.exists(CUSTOM_PROMPT_FILE):
            with open(CUSTOM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('custom_prompt', ''), data.get('enabled', False)
        return '', False
    except Exception as e:
        log(f"Error loading custom prompt: {e}", "warning")
        return '', False

def save_custom_prompt(custom_prompt, enabled):
    try:
        data = {
            'custom_prompt': custom_prompt,
            'enabled': enabled,
            'updated_at': datetime.now().isoformat()
        }
        with open(CUSTOM_PROMPT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log(f"Error saving custom prompt: {e}", "error")
        return False

def generate_reply(query, email_user, openai_api_key):
    try:
        if not query.strip(): 
            return "Error: Empty query provided"
        
        if not openai_api_key or not openai_api_key.startswith('sk-'):
            return "Error: Valid OpenAI API key required"
        
        # Set OpenAI API key
        openai.api_key = openai_api_key
        
        # Load custom prompt
        custom_prompt, custom_enabled = load_custom_prompt()
        
        # Extract keywords for context
        query_keywords = extract_keywords(query)
        keyword_context = f"Keywords found: {', '.join(query_keywords)}" if query_keywords else ""
        
        base_prompt = "You are a professional customer service agent. Generate helpful, concise, and polite replies. Keep responses under 200 words."
        system_prompt = f"{base_prompt}\n\nAdditional Instructions: {custom_prompt.strip()}" if custom_enabled and custom_prompt.strip() else base_prompt
        
        user_content = f"{keyword_context}\n\nCustomer Email: {query[:500]}\n\nGenerate a professional reply:"
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=250, 
            temperature=0.7, 
            timeout=30
        )
        return response.choices[0].message.content.strip()
        
    except openai.error.RateLimitError:
        return "I apologize for the delay. We are experiencing high volume. Please allow us some time to respond to your inquiry personally."
    except openai.error.AuthenticationError:
        return "Error: Invalid OpenAI API key. Please check your API key."
    except Exception as e:
        log(f"Error generating reply: {str(e)}", "error")
        return "Thank you for contacting us. We will review your message and respond as soon as possible."

# Storage Functions
def save_data(data, file):
    try:
        if file.endswith('.csv'):
            if not data:
                pd.DataFrame(columns=['subject','from','date','customer','reply','has_reply']).to_csv(file, index=False)
            else:
                pd.DataFrame(data).to_csv(file, index=False)
        else:
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        error_msg = f"Error saving data to {file}: {str(e)}"
        log(error_msg, "error")
        return False

def load_data(file):
    try:
        if not os.path.exists(file):
            return [] if file.endswith('.csv') else {}
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            return df.fillna('').to_dict('records') if not df.empty else []
        else:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception as e:
        error_msg = f"Error loading {file}: {str(e)}"
        log(error_msg, "error")
        return [] if file.endswith('.csv') else {}

# UI Functions
def display_metrics(data):
    try:
        total = len(data) if isinstance(data, list) else data.get('total', 0)
        with_replies = len([e for e in data if e.get('has_reply', False)]) if isinstance(data, list) else data.get('with_replies', 0)
        pending = total - with_replies
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("📧 Total Emails", total),
            ("✅ With Replies", with_replies),
            ("⏳ Pending", pending),
            ("📊 Loaded", total)
        ]
        for col, (title, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f'<div class="metric-card"><h3>{title}</h3><h2>{value}</h2></div>', 
                          unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def search_and_load_emails(email_user, email_pass, email_limit, folder):
    try:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔐 Authenticating with Gmail...")
            progress_bar.progress(20)
            
            mail = login_gmail(email_user, email_pass)
            
            def update_progress(percent, message):
                progress_bar.progress(20 + (percent * 0.6) / 100)
                status_text.text(f"🔄 {message}")
            
            if folder == "both":
                inbox_emails = fetch_emails(mail, 'inbox', email_limit//2, update_progress)
                sent_emails = fetch_emails(mail, 'sent', email_limit//2, update_progress)
                all_emails = inbox_emails + sent_emails
                pairs = match_replies(inbox_emails, sent_emails)
            else:
                all_emails = fetch_emails(mail, folder, email_limit, update_progress)
                pairs = []
            
            mail.logout()
            
            status_text.text("💾 Saving data...")
            progress_bar.progress(90)
            
            if save_data(pairs if pairs else all_emails, CSV_FILE):
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                st.success(f"🎉 Successfully loaded {len(all_emails)} emails!")
                if pairs:
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("📧 Inbox", len(inbox_emails))
                    with col2: st.metric("📤 Sent", len(sent_emails))
                    with col3: st.metric("🔗 Pairs", len(pairs))
                    display_metrics(pairs)
                    st.session_state['email_pairs'] = pairs
                else:
                    display_metrics(all_emails)
                    st.session_state['emails'] = all_emails
                
                time.sleep(2)
                progress_container.empty()
                st.rerun()
            else:
                st.error("❌ Failed to save email data")
    except Exception as e:
        error_msg = f"Search and load error: {str(e)}"
        st.error(f"❌ Error: {error_msg}")
        log(error_msg, "error")

def reply_management_section(email_user, email_pass, openai_api_key):
    try:
        # Try to load from pairs first, then emails
        email_pairs = st.session_state.get('email_pairs', [])
        emails = st.session_state.get('emails', [])
        
        if not email_pairs and not emails:
            emails = load_data(CSV_FILE)
        
        all_data = email_pairs if email_pairs else emails
        
        if not all_data:
            st.info("📭 No emails loaded. Please use the 'Search & Load' tab first.")
            return
        
        st.subheader("🔍 Find Emails to Reply")
        
        # Filter emails that need replies
        if email_pairs:
            pending = [e for e in email_pairs if not e.get('has_reply', False) and str(e.get('customer', '')).strip()]
        else:
            pending = [e for e in emails if str(e.get('body', '')).strip()]
        
        if not pending:
            st.success("🎉 All emails have been processed!")
            return

        st.write(f"📬 Found **{len(pending)}** emails needing attention")
        
        # Display emails for selection
        if 'selected_emails' not in st.session_state:
            st.session_state.selected_emails = []

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Select All"):
                st.session_state.selected_emails = pending.copy()
                st.rerun()
        with col2:
            if st.button("❌ Clear Selection"):
                st.session_state.selected_emails = []
                st.rerun()
        with col3:
            st.write(f"Selected: **{len(st.session_state.selected_emails)}**")

        # Show first 10 emails
        for i, email in enumerate(pending[:10]):
            col1, col2 = st.columns([0.5, 9.5])
            with col1:
                selected = st.checkbox("", key=f"select_{i}")
                if selected and email not in st.session_state.selected_emails:
                    st.session_state.selected_emails.append(email)
                elif not selected and email in st.session_state.selected_emails:
                    st.session_state.selected_emails.remove(email)
            with col2:
                from_field = str(email.get('from', 'Unknown'))[:50]
                subject_field = str(email.get('subject', 'No Subject'))[:60]
                with st.expander(f"📧 **{from_field}** | {subject_field}"):
                    st.write(f"**From:** {from_field}")
                    st.write(f"**Subject:** {subject_field}")
                    st.write(f"**Date:** {str(email.get('date',''))[:25]}")
                    
                    content_text = str(email.get('customer', email.get('body', '')))
                    preview_text = content_text[:300] + ("..." if len(content_text) > 300 else "")
                    st.text_area("Content:", preview_text, height=100, disabled=True, key=f"preview_{i}")
        
        if len(pending) > 10:
            st.info(f"📄 Showing 10 of {len(pending)} emails. Select emails above to generate replies.")
        
        # Generate replies section
        if st.session_state.selected_emails:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🤖 Generate {len(st.session_state.selected_emails)} Replies", type="primary"):
                    generate_ai_replies(st.session_state.selected_emails, email_user, openai_api_key)
            with col2:
                if st.button("🗑️ Clear Selected"):
                    st.session_state.selected_emails = []
                    st.rerun()

        # Display generated replies
        if 'generated_replies' in st.session_state and st.session_state.generated_replies:
            display_generated_replies(email_user, email_pass, openai_api_key)
    
    except Exception as e:
        st.error(f"❌ Reply management error: {str(e)}")
        log(f"Reply management error: {str(e)}", "error")

def generate_ai_replies(selected_emails, email_user, openai_api_key):
    try:
        if not openai_api_key or not openai_api_key.startswith('sk-'):
            st.error("❌ OpenAI API key not configured or invalid")
            return
        
        st.session_state.generated_replies = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, email in enumerate(selected_emails):
            status_text.text(f"🤖 Generating reply {idx+1} of {len(selected_emails)}...")
            
            # Get email content
            customer_text = str(email.get('customer', email.get('body', ''))).strip()
            if customer_text:
                reply = generate_reply(customer_text, email_user, openai_api_key)
                st.session_state.generated_replies.append({
                    'email': email,
                    'draft': reply,
                    'status': 'draft'
                })
            progress_bar.progress((idx+1)/len(selected_emails))
        
        status_text.text("✅ Reply generation complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"🎉 Generated {len(st.session_state.generated_replies)} replies!")
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error generating replies: {str(e)}")
        log(f"Error generating replies: {str(e)}", "error")

def display_generated_replies(email_user, email_pass, openai_api_key):
    try:
        st.divider()
        st.subheader("📝 Generated Replies - Review & Send")
        
        for i, reply_item in enumerate(st.session_state.generated_replies):
            email = reply_item['email']
            draft = reply_item['draft']
            status = reply_item.get('status','draft')
            status_icon = "✅" if status=='sent' else "❌" if status=='error' else "📝"
            subject = str(email.get('subject','No Subject'))[:50]
            
            with st.expander(f"{status_icon} Reply {i+1}: {subject}...", expanded=(i<3)):
                col1, col2 = st.columns([1,2])
                with col1:
                    st.write(f"**To:** {str(email.get('from','Unknown'))[:40]}...")
                    st.write(f"**Subject:** {subject}...")
                    st.write(f"**Status:** {status.title()}")
                    
                    # Extract recipient email
                    recipient_match = re.search(r'<(.+?)>', str(email.get('from','')))
                    recipient_email = recipient_match.group(1) if recipient_match else str(email.get('from',''))
                
                with col2:
                    st.write("**Original Email:**")
                    original_text = str(email.get('customer', email.get('body', '')))[:300]
                    st.text_area("Original:", original_text + ("..." if len(str(email.get('customer', email.get('body', ''))))>300 else ""), 
                               height=80, disabled=True, key=f"orig_{i}")
                
                st.write("**Generated Reply:**")
                edited_reply = st.text_area("Edit your reply:", draft, height=120, key=f"reply_edit_{i}")
                
                if status != 'sent':
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("📧 Send Email", key=f"send_{i}", type="primary"):
                            success, message = send_email(email_user, email_pass, recipient_email, subject, edited_reply)
                            if success:
                                st.success(f"✅ {message}")
                                st.session_state.generated_replies[i]['status'] = 'sent'
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    
                    with c2:
                        if st.button("🔄 Regenerate", key=f"regen_{i}"):
                            original_content = str(email.get('customer', email.get('body', ''))).strip()
                            new_reply = generate_reply(original_content, email_user, openai_api_key)
                            st.session_state.generated_replies[i]['draft'] = new_reply
                            st.session_state.generated_replies[i]['status'] = 'draft'
                            st.success("✅ Reply regenerated!")
                            st.rerun()
                    
                    with c3:
                        if st.button("💾 Save Draft", key=f"save_{i}"):
                            try:
                                drafts = load_data('drafts.json') if os.path.exists('drafts.json') else []
                                drafts.append({
                                    'subject': subject,
                                    'from': str(email.get('from','')),
                                    'reply': edited_reply,
                                    'saved_at': datetime.now().isoformat(),
                                    'original_email': original_text
                                })
                                if save_data(drafts, 'drafts.json'):
                                    st.success("💾 Draft saved!")
                                else:
                                    st.error("❌ Failed to save draft")
                            except Exception as e:
                                st.error(f"❌ Save error: {str(e)}")
                    
                    with c4:
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.generated_replies.pop(i)
                            st.success("🗑️ Reply removed!")
                            st.rerun()
                else:
                    st.success("✅ Email sent successfully!")
        
        # Bulk actions
        if st.session_state.generated_replies:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Replies"):
                    st.session_state.generated_replies = []
                    st.success("🗑️ All replies cleared!")
                    st.rerun()
            with col2:
                pending_count = len([r for r in st.session_state.generated_replies if r.get('status') != 'sent'])
                if pending_count > 0:
                    st.write(f"📋 {pending_count} replies pending")
    
    except Exception as e:
        st.error(f"❌ Error displaying replies: {str(e)}")
        log(f"Error displaying replies: {str(e)}", "error")

def display_analytics():
    try:
        # Try different data sources
        email_pairs = st.session_state.get('email_pairs', [])
        emails = st.session_state.get('emails', [])
        
        if not email_pairs and not emails:
            emails = load_data(CSV_FILE)
        
        data = email_pairs if email_pairs else emails
        
        if not data:
            st.info("📭 No email data available. Load emails first to see analytics.")
            return
        
        st.subheader("📊 Email Overview")
        display_metrics(data)
        
        # Keywords analysis
        st.subheader("🏷️ Common Keywords")
        all_keywords = []
        for e in data:
            content = str(e.get('customer', e.get('body', '')))
            all_keywords.extend(extract_keywords(content))
        
        if all_keywords:
            from collections import Counter
            keyword_counts = Counter(all_keywords).most_common(10)
            cols = st.columns(min(5, len(keyword_counts)))
            for i, (kw, count) in enumerate(keyword_counts[:5]):
                with cols[i]:
                    st.metric(f"#{i+1} {kw.title()}", count)
            
            if len(keyword_counts) > 5:
                with st.expander("View All Keywords"):
                    for kw, count in keyword_counts:
                        st.write(f"**{kw.title()}**: {count} emails")
        
        # Response statistics
        st.subheader("📈 Email Statistics")
        total_emails = len(data)
        
        if email_pairs:
            replied = len([e for e in data if e.get('has_reply', False)])
            c1, c2, c3 = st.columns(3)
            with c1:
                rate = (replied/total_emails*100) if total_emails > 0 else 0
                st.metric("Response Rate", f"{rate:.1f}%")
            with c2:
                st.metric("Total Conversations", total_emails)
            with c3:
                st.metric("Pending Responses", total_emails - replied)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Emails", total_emails)
            with c2:
                st.metric("Processed Today", total_emails)
            with c3:
                st.metric("Success Rate", "100%")
        
        # Email distribution chart
        if len(data) > 0:
            st.subheader("📊 Email Distribution")
            
            # Create a simple chart based on email dates or subjects
            chart_data = []
            for email in data[:20]:  # Limit for performance
                subject = str(email.get('subject', 'No Subject'))[:30]
                chart_data.append({'Subject': subject, 'Count': 1})
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                df_grouped = df.groupby('Subject').sum().reset_index()
                st.bar_chart(df_grouped.set_index('Subject'))
    
    except Exception as e:
        st.error(f"❌ Analytics error: {str(e)}")
        log(f"Analytics error: {str(e)}", "error")

def system_settings():
    try:
        st.subheader("🔧 System Maintenance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**System Information:**")
            st.write(f"Python Version: {sys.version[:10]}")
            st.write(f"Streamlit Version: {st.__version__}")
            st.write(f"Google Drive Available: {'Yes' if GOOGLE_DRIVE_AVAILABLE else 'No'}")
            
            st.write("**Local Data Files:**")
            for fn in [CSV_FILE, 'drafts.json', LOG_FILE, CUSTOM_PROMPT_FILE]:
                try:
                    if os.path.exists(fn):
                        size_mb = os.path.getsize(fn) / (1024 * 1024)
                        st.write(f"✅ {fn}: {size_mb:.2f} MB")
                    else:
                        st.write(f"❌ {fn}: Not found")
                except Exception as e:
                    st.write(f"⚠️ {fn}: Error reading")
        
        with col2:
            st.write("**Actions:**")
            if st.button("🗑️ Clear Session Data"):
                # Clear session state but keep credentials
                for key in list(st.session_state.keys()):
                    if key not in ['email_user', 'email_pass', 'openai_api_key']:
                        del st.session_state[key]
                st.success("✅ Session data cleared!")
                st.rerun()
            
            if st.button("📊 Export Data"):
                try:
                    export_data = {}
                    
                    # Include session data
                    if 'emails' in st.session_state:
                        export_data['emails'] = st.session_state['emails']
                    if 'email_pairs' in st.session_state:
                        export_data['email_pairs'] = st.session_state['email_pairs']
                    
                    # Include file data
                    for fn, key in [(CSV_FILE, 'csv_data'), ('drafts.json', 'drafts'), (CUSTOM_PROMPT_FILE, 'custom_prompt')]:
                        if os.path.exists(fn):
                            export_data[key] = load_data(fn)
                    
                    export_data['exported_at'] = datetime.now().isoformat()
                    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        "📥 Download Data Export",
                        data=export_json,
                        file_name=f"email_assistant_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")
        
        # Custom Prompts
        st.divider()
        st.subheader("🎯 Custom AI Prompt Settings")
        
        try:
            custom_prompt, custom_enabled = load_custom_prompt()
            
            st.markdown("""
            <div class="custom-prompt-box">
                <h4>🤖 Customize AI Reply Generation</h4>
                <p>Add custom instructions to personalize how the AI generates email replies.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_custom_prompt = st.text_area(
                    "Additional AI Instructions:",
                    value=custom_prompt,
                    height=150,
                    placeholder="Example: Always include our company phone number (555-123-4567). Use a friendly tone. Mention our 30-day return policy when relevant."
                )
            with col2:
                status_color = "🟢" if custom_enabled else "🔴"
                st.write(f"{status_color} {'Enabled' if custom_enabled else 'Disabled'}")
                if custom_prompt.strip():
                    wc = len(custom_prompt.split())
                    st.write(f"📝 Words: {wc}")
                    if wc > 100:
                        st.warning("⚠️ Long prompts may affect performance")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("✅ Enable"):
                    if new_custom_prompt.strip() and save_custom_prompt(new_custom_prompt, True):
                        st.success("✅ Custom prompt enabled!")
                        st.rerun()
            with c2:
                if st.button("❌ Disable"):
                    if save_custom_prompt(new_custom_prompt, False):
                        st.success("❌ Custom prompt disabled!")
                        st.rerun()
            with c3:
                if st.button("💾 Save"):
                    if save_custom_prompt(new_custom_prompt, custom_enabled):
                        st.success("💾 Prompt saved!")
                        st.rerun()
            with c4:
                if st.button("🗑️ Clear"):
                    if save_custom_prompt("", False):
                        st.success("🗑️ Prompt cleared!")
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Custom prompt error: {str(e)}")
        
        # Google Drive Integration
        st.divider()
        st.subheader("☁️ Google Drive Integration")
        
        if not GOOGLE_DRIVE_AVAILABLE:
            st.error("❌ Google Drive libraries not installed")
            st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        else:
            st.success("✅ Google Drive libraries available")
            
            # Simple file upload for credentials
            uploaded_file = st.file_uploader(
                "Upload Google API Credentials JSON",
                type=['json'],
                help="Download from Google Cloud Console"
            )
            
            if uploaded_file is not None:
                try:
                    credentials_data = json.load(uploaded_file)
                    with open(DRIVE_CREDS_FILE, 'w') as f:
                        json.dump(credentials_data, f, indent=2)
                    st.success("✅ Credentials file saved!")
                    st.info("💡 Full Google Drive integration can be configured for production use.")
                except Exception as e:
                    st.error(f"❌ Invalid credentials file: {str(e)}")
            
            # Show status
            if os.path.exists(DRIVE_CREDS_FILE):
                st.success("✅ Google Drive credentials configured")
            else:
                st.warning("⚠️ Google Drive credentials not configured")
    
    except Exception as e:
        st.error(f"❌ System settings error: {str(e)}")
        log(f"System settings error: {str(e)}", "error")

def main():
    try:
        load_custom_css()
        
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>🤖 AI Email Assistant</h1>
            <p style="font-size: 1.2rem; color: #666;">Streamline customer support with AI-powered email management</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Debug info
            st.write("🔍 **Debug Info:**")
            st.write(f"Python: {sys.version[:5]}")
            st.write(f"Streamlit: {st.__version__}")
            st.write(f"Google Drive: {'✅' if GOOGLE_DRIVE_AVAILABLE else '❌'}")
            
            # OpenAI API Key Input
            with st.expander("🤖 OpenAI API Settings", expanded=True):
                try:
                    openai_api_key = st.text_input(
                        "🔑 OpenAI API Key", 
                        type='password',
                        placeholder="sk-...",
                        help="Enter your OpenAI API key to enable AI reply generation"
                    )
                    if openai_api_key:
                        if openai_api_key.startswith('sk-'):
                            st.session_state['openai_api_key'] = openai_api_key
                            st.success("✅ API key configured")
                        else:
                            st.error("❌ Invalid API key format")
                    else:
                        st.warning("⚠️ API key required for AI features")
                except Exception as e:
                    st.error(f"❌ OpenAI config error: {str(e)}")
            
            # Email Settings
            with st.expander("📧 Email Settings", expanded=True):
                try:
                    email_user = st.text_input("📧 Gmail Address", placeholder="your-email@gmail.com")
                    email_pass = st.text_input("🔑 Gmail App Password", type='password')
                    if email_user and email_pass:
                        st.session_state.update({'email_user': email_user, 'email_pass': email_pass})
                except Exception as e:
                    st.error(f"❌ Email config error: {str(e)}")
            
            # System Status
            st.header("📊 System Status")
            try:
                csv_exists = os.path.exists(CSV_FILE)
                openai_configured = st.session_state.get('openai_api_key', '').startswith('sk-')
                
                st.write("**Configuration:**")
                st.write(f"📧 Email: {'✅' if st.session_state.get('email_user') else '❌'}")
                st.write(f"🤖 OpenAI: {'✅' if openai_configured else '❌'}")
                st.write(f"☁️ Google Drive: {'✅' if GOOGLE_DRIVE_AVAILABLE else '❌'}")
                st.write(f"📁 Data File: {'✅' if csv_exists else '❌'}")
                
                # Session data status
                emails_count = len(st.session_state.get('emails', []))
                pairs_count = len(st.session_state.get('email_pairs', []))
                if emails_count > 0 or pairs_count > 0:
                    st.write(f"💾 Session: {emails_count + pairs_count} emails")
            except Exception as e:
                st.error(f"❌ Status error: {str(e)}")

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Load", "✍️ Reply Management", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            try:
                st.header("🔍 Email Search & Loading")
                
                email_user = st.session_state.get('email_user', '')
                email_pass = st.session_state.get('email_pass', '')
                
                if not email_user or not email_pass:
                    st.warning("⚠️ Please configure Gmail credentials in the sidebar first.")
                    return
                
                # Search controls
                col1, col2, col3, col4 = st.columns(4)
                with col1: 
                    email_limit = st.slider("📧 Max Emails:", 10, 200, 50, step=10)
                with col2: 
                    folder = st.selectbox("📁 Folder:", ["inbox", "sent", "both"])
                with col3: 
                    search_type = st.selectbox("🔍 Search:", ["all", "recent", "today"])
                with col4:
                    load_mode = st.selectbox("🔄 Mode:", ["standard", "with_replies"])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Load Emails", type="primary", use_container_width=True):
                        search_and_load_emails(email_user, email_pass, email_limit, folder)
                
                with col2:
                    if st.button("📊 Quick Preview", use_container_width=True):
                        try:
                            with st.spinner("Getting preview..."):
                                mail = login_gmail(email_user, email_pass)
                                preview_emails = fetch_emails(mail, 'inbox', 5)
                                mail.logout()
                            
                            if preview_emails:
                                st.success(f"✅ Preview: {len(preview_emails)} recent emails")
                                for email in preview_emails:
                                    st.write(f"📧 **{email.get('subject', 'No Subject')[:50]}** from {email.get('from', 'Unknown')[:30]}")
                            else:
                                st.warning("No emails found")
                        except Exception as e:
                            st.error(f"❌ Preview error: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Search & Load error: {str(e)}")
                log(f"Search & Load error: {str(e)}", "error")

        with tab2:
            try:
                st.header("✍️ Email Reply Management")
                
                openai_api_key = st.session_state.get('openai_api_key', '')
                email_user = st.session_state.get('email_user', '')
                email_pass = st.session_state.get('email_pass', '')
                
                if not openai_api_key or not openai_api_key.startswith('sk-'):
                    st.warning("⚠️ Please configure OpenAI API key in the sidebar first.")
                    return
                
                if not email_user or not email_pass:
                    st.warning("⚠️ Please configure Gmail credentials in the sidebar first.")
                    return
                
                reply_management_section(email_user, email_pass, openai_api_key)
            
            except Exception as e:
                st.error(f"❌ Reply Management error: {str(e)}")
                log(f"Reply Management error: {str(e)}", "error")

        with tab3:
            try:
                st.header("📊 Email Analytics")
                display_analytics()
            except Exception as e:
                st.error(f"❌ Analytics error: {str(e)}")
                log(f"Analytics error: {str(e)}", "error")

        with tab4:
            try:
                st.header("⚙️ System Settings")
                system_settings()
            except Exception as e:
                st.error(f"❌ Settings error: {str(e)}")
                log(f"Settings error: {str(e)}", "error")
    
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.code(traceback.format_exc())
        log(f"Critical error: {str(e)}", "error")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"❌ Fatal application error: {str(e)}")
        st.code(traceback.format_exc())
        print(f"Fatal error: {str(e)}")
        print(traceback.format_exc())