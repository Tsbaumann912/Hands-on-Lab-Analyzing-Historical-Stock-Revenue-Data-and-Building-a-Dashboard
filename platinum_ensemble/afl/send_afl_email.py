#!/usr/bin/env python3
"""Email PlatinumEnsemble_CTA.afl via Gmail SMTP (requires app password).

  export GMAIL_ADDRESS=you@gmail.com
  export GMAIL_APP_PASSWORD=xxxx
  python3 send_afl_email.py Tsbaumann912@gmail.com
"""
from __future__ import annotations
import os, sys, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

def main() -> int:
    to_addr = sys.argv[1] if len(sys.argv) > 1 else "Tsbaumann912@gmail.com"
    user = os.environ.get("GMAIL_ADDRESS") or os.environ.get("SMTP_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD") or os.environ.get("SMTP_PASSWORD")
    if not user or not password:
        print("Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD (Gmail app password).", file=sys.stderr)
        return 2
    afl = Path(__file__).with_name("PlatinumEnsemble_CTA.afl")
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = "PlatinumEnsemble CTA — AmiBroker AFL file"
    msg.attach(MIMEText(
        "Attached: PlatinumEnsemble_CTA.afl (AmiBroker port of the locked platinum ensemble).\n"
        "Open in AmiBroker Formula Editor. Needs GC and DX in the database.\n",
        "plain",
    ))
    part = MIMEBase("application", "octet-stream")
    part.set_payload(afl.read_bytes())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", 'attachment; filename="PlatinumEnsemble_CTA.afl"')
    msg.attach(part)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as s:
        s.starttls()
        s.login(user, password)
        s.sendmail(user, [to_addr], msg.as_string())
    print(f"Sent {afl.name} to {to_addr}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
