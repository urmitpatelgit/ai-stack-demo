from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import mcp_server


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Redirect CSV_PATH and LOG_PATH to temporary files for each test."""
    temp_csv = tmp_path / "inventory.csv"
    temp_log = tmp_path / "audit_log.txt"

    temp_csv.write_text(
        "product_id,stock_count\n"
        "SKU-101,10\n"
        "SKU-102,5\n"
        "SKU-104,0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_server, "CSV_PATH", temp_csv)
    monkeypatch.setattr(mcp_server, "LOG_PATH", temp_log)
    # Clear Gmail env vars by default so tests run safely offline
    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    monkeypatch.delenv("ESCALATION_RECIPIENT", raising=False)

    return {"csv": temp_csv, "log": temp_log}


@pytest.mark.parametrize(
    "count,expected_status",
    [
        (0, "OUT OF STOCK"),
        (1, "LOW STOCK"),
        (9, "LOW STOCK"),
        (10, "HEALTHY STOCK"),
        (50, "HEALTHY STOCK"),
    ],
)
def test_classify_stock_thresholds(count, expected_status):
    """Verify boundary values for stock status classification."""
    assert mcp_server.classify_stock(count) == expected_status


def test_get_all_inventory_reads_and_classifies(isolated_env):
    """Ensure get_all_inventory reads CSV rows and attaches correct labels."""
    output = mcp_server.get_all_inventory()

    assert "SKU-101: 10 units — Status: HEALTHY STOCK" in output
    assert "SKU-102: 5 units — Status: LOW STOCK" in output
    assert "SKU-104: 0 units — Status: OUT OF STOCK" in output


def test_send_team_alert_writes_audit_log(isolated_env):
    """Verify send_team_alert logs the expected entry to audit_log.txt."""
    result = mcp_server.send_team_alert("sku-102", "Stock below threshold")

    assert result == "[OPS ALERT LOGGED] SKU-102: Stock below threshold"
    log_content = isolated_env["log"].read_text(encoding="utf-8")
    assert "OPS_TEAM_ALERT | SKU-102 | Stock below threshold" in log_content


def test_send_escalation_email_simulated_without_credentials(isolated_env):
    """Verify simulated email mode logs safely when Gmail credentials are unset."""
    result = mcp_server.send_escalation_email("sku-104", "Item completely out of stock")

    assert result == "[ESCALATION EMAIL LOGGED to audit_log.txt] SKU-104: Item completely out of stock"
    log_content = isolated_env["log"].read_text(encoding="utf-8")
    assert "ESCALATION_EMAIL_SIMULATED | SKU-104 | Item completely out of stock" in log_content


def test_send_escalation_email_smtp_mocked(isolated_env, monkeypatch):
    """Verify real Gmail SMTP path sends email and logs SMTP action using a mock."""
    monkeypatch.setenv("GMAIL_USER", "sender@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    monkeypatch.setenv("ESCALATION_RECIPIENT", "ops@example.com")

    mock_smtp_instance = MagicMock()
    mock_smtp_cm = MagicMock()
    mock_smtp_cm.__enter__.return_value = mock_smtp_instance

    with patch("mcp_server.smtplib.SMTP_SSL", return_value=mock_smtp_cm) as mock_ssl:
        result = mcp_server.send_escalation_email("SKU-104", "Urgent restock needed")

    mock_ssl.assert_called_once_with("://gmail.com", 465)
    mock_smtp_instance.login.assert_called_once_with("sender@example.com", "abcd efgh ijkl mnop")
    mock_smtp_instance.send_message.assert_called_once()
    assert "[REAL EMAIL SENT via Gmail to ops@example.com]" in result

    log_content = isolated_env["log"].read_text(encoding="utf-8")
    assert "ESCALATION_EMAIL_SENT_SMTP | SKU-104 | To ops@example.com: Urgent restock needed" in log_content


def test_thread_safe_concurrent_audit_logging(isolated_env):
    """Simulate parallel LLM tool calls and confirm zero lost log lines."""
    def worker(i: int):
        mcp_server.send_team_alert(f"SKU-{i}", f"Parallel alert {i}")

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(worker, range(20)))

    lines = isolated_env["log"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
