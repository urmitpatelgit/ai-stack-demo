import csv
import os
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Expose inventory and escalation operations as MCP tools over stdio.
mcp = FastMCP("OperationsServer")

# Resolve data files relative to this module so the server is independent of the
# process working directory.
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "inventory.csv"
LOG_PATH = BASE_DIR / "audit_log.txt"

# Serialize audit writes because MCP tools may run in separate threads.
LOG_LOCK = threading.Lock()


def write_audit_log(action_type: str, product_id: str, details: str) -> None:
    """Thread-safe append of a timestamped record to audit_log.txt."""
    # Millisecond precision keeps related operations easy to order in the log.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with LOG_LOCK:
        with open(LOG_PATH, mode="a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {action_type} | {product_id} | {details}\n")
            log_file.flush()


def classify_stock(count: int) -> str:
    # Keep the stock labels centralized so inventory output stays consistent.
    if count == 0:
        return "OUT OF STOCK"
    if count < 10:
        return "LOW STOCK"
    return "HEALTHY STOCK"


@mcp.tool()
def get_all_inventory() -> str:
    """Read all products and their current stock status from inventory.csv."""
    rows = []
    with open(CSV_PATH, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Normalize SKUs before exposing or logging them.
            sku = row["product_id"].strip().upper()
            count = int(row["stock_count"])
            status = classify_stock(count)
            rows.append(f"{sku}: {count} units — Status: {status}")
    return "\n".join(rows) if rows else "inventory.csv is empty."


@mcp.tool()
def send_team_alert(product_id: str, message: str) -> str:
    """Send a low-stock alert to the operations team and record it in audit_log.txt."""
    sku = product_id.strip().upper()
    # This implementation records an alert in the shared audit trail.
    write_audit_log("OPS_TEAM_ALERT", sku, message)
    return f"[OPS ALERT LOGGED] {sku}: {message}"


@mcp.tool()
def send_escalation_email(product_id: str, message: str) -> str:
    """Send an urgent escalation email (via Gmail SMTP if configured) and log to audit_log.txt."""
    sku = product_id.strip().upper()

    # SMTP credentials and the recipient are environment-based configuration,
    # allowing local development without exposing secrets in source.
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("ESCALATION_RECIPIENT", gmail_user)

    if gmail_user and gmail_pass and recipient:
        # Build a complete email object before opening the SMTP connection.
        msg = EmailMessage()
        msg["Subject"] = f"URGENT: Out-of-Stock Escalation for {sku}"
        msg["From"] = gmail_user
        msg["To"] = recipient
        msg.set_content(message)

        with smtplib.SMTP_SSL("://gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pass)
            smtp.send_message(msg)

        write_audit_log("ESCALATION_EMAIL_SENT_SMTP", sku, f"To {recipient}: {message}")
        return f"[REAL EMAIL SENT via Gmail to {recipient}] {sku}: {message}"

    # When SMTP is not configured, preserve a safe local audit-only behavior.
    write_audit_log("ESCALATION_EMAIL_SIMULATED", sku, message)
    return f"[ESCALATION EMAIL LOGGED to audit_log.txt] {sku}: {message}"


if __name__ == "__main__":
    # stdio lets an MCP host start and communicate with this server as a process.
    mcp.run(transport="stdio")
