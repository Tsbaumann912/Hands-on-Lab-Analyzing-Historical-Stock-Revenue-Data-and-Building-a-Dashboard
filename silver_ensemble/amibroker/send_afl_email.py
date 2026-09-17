#!/usr/bin/env python3
"""Send SilverEnsemble_SI.afl via Gmail SMTP.

Requires env:
  GMAIL_USER          e.g. you@gmail.com
  GMAIL_APP_PASSWORD  Google App Password (not account password)
  EMAIL_TO            default Tsbaumann912@gmail.com
"""

from __future__ import annotations

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def main() -> int:
    user = os.environ.get("GMAIL_USER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    to_addr = os.environ.get("EMAIL_TO", "Tsbaumann912@gmail.com").strip()
    afl = Path(__file__).resolve().parent / "SilverEnsemble_SI.afl"
    if not user or not password:
        print("Missing GMAIL_USER / GMAIL_APP_PASSWORD", file=sys.stderr)
        return 2
    if not afl.is_file():
        print(f"Missing AFL file: {afl}", file=sys.stderr)
        return 2

    msg = EmailMessage()
    msg["Subject"] = "SilverEnsemble SI — AmiBroker AFL strategy file"
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(
        "Attached is SilverEnsemble_SI.afl — AmiBroker AFL port of the Python "
        "silver_ensemble COMEX Silver CTA.\n\n"
        "Install: Formula Editor → open file → Tools → Send to Analysis.\n"
        "Analysis Settings → Positions = Long and Short.\n"
    )
    msg.add_attachment(
        afl.read_bytes(),
        maintype="text",
        subtype="plain",
        filename="SilverEnsemble_SI.afl",
    )

    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as smtp:
        smtp.starttls(context=ctx)
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"Sent {afl.name} to {to_addr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
