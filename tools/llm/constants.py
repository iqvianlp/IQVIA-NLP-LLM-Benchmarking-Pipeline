# This file contains all the constants for AA's OPEN AI account
import os
from pathlib import Path

from dotenv import load_dotenv

# load env variables from .env if any
load_dotenv(Path(__file__).parent / "azure.env")

# Azure Cloud
TENANT_ID = os.environ.get("TENANT_ID", "")

# Client secrets
NON_INTERACTIVE_CLIENT_ID = os.environ.get("NON_INTERACTIVE_CLIENT_ID", "")
SERVICE_PRINCIPAL = os.environ.get("SERVICE_PRINCIPAL", "")
SERVICE_PRINCIPAL_SECRET = os.environ.get("SERVICE_PRINCIPAL_SECRET", "")
SCOPE_NON_INTERACTIVE = f"api://{NON_INTERACTIVE_CLIENT_ID}/.default"

# OpenAI
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2023-03-15-preview")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
