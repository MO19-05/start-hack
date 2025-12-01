import smtplib
from email.message import EmailMessage
from typing import List, Optional
import asyncio
from .config import settings


def _parse_recipients(recipients: Optional[str]) -> List[str]:
    if not recipients:
        return []
    return [r.strip() for r in recipients.split(",") if r.strip()]


def send_email_sync(subject: str, body: str, recipients: Optional[List[str]] = None) -> bool:
    """Send email synchronously using blocking smtplib. Returns True on success."""
    recipients_list = recipients or _parse_recipients(settings.ALERT_RECIPIENTS)
    if not recipients_list:
        print("No recipients configured for email; skipping send")
        return False

    if not settings.SMTP_HOST or not settings.SMTP_PORT:
        print("SMTP not configured; skipping send")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM or (settings.SMTP_USER or "noreply@example.com")
    msg["To"] = ", ".join(recipients_list)
    msg.set_content(body)

    try:
        if settings.SMTP_PORT == 465:
            smtp = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10)
        else:
            smtp = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10)
            if settings.SMTP_USE_TLS:
                smtp.starttls()

        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)

        smtp.send_message(msg)
        smtp.quit()
        print("Email sent to", recipients_list)
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False


async def send_email_async(subject: str, body: str, recipients: Optional[List[str]] = None) -> bool:
    return await asyncio.to_thread(send_email_sync, subject, body, recipients)
