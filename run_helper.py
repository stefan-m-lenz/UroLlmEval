import traceback
import smtplib
import os
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_notification_email(subject, body=""):
    """
    Sends an email with the given subject and body using an SMTP mail relay server.
    The SMTP server details and credentials are read from environment variables.
    """

    # Fetch SMTP configuration from environment variables
    smtp_server = os.getenv("UROLLMEVAL_MAIL_SERVER")
    smtp_port = int(os.getenv("UROLLMEVAL_MAIL_PORT", 25))
    from_email = os.getenv("UROLLMEVAL_FROM_EMAIL")
    to_email = os.getenv("UROLLMEVAL_TO_EMAIL")

    # If any critical configuration is missing, do nothing
    if not all([smtp_server, smtp_port, from_email, to_email]):
        print(f'(SMTP configuration is incomplete. Cannot send email notification with subject: "{subject}")')
        return

    # Construct the email message
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Attach the HTML body
    msg.attach(MIMEText(body, "html"))

    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to secure
            #server.login(smtp_username, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())
    except Exception as e:
        print(f"Failed to send email due to an error: {e}")


def send_exception_email(exception: Exception):
    """
    Sends an email with the details of an exception using an SMTP mail relay server.
    """

    # Create the subject and body for the email
    subject = f"Exception occurred running UroLlmEval: {type(exception).__name__}"
    body = f"<pre>{traceback.format_exc()}</pre>"

    send_notification_email(subject=subject, body=body)


def run_script(script_name, **kwargs):
    """
    Run a Python script as a subprocess with the provided keyword arguments as command-line arguments.
    """
    # Construct the command
    cmd = [
        "python",
        script_name,
    ]

    # Add arguments as key-value pairs
    for key, value in kwargs.items():
        if value is not None:  # Only include arguments with values
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    try:
        # Run the command after constructing the full `cmd` list
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_name} with arguments {kwargs}: {e}")
        print(f"Command that failed: {' '.join(cmd)}")
        raise