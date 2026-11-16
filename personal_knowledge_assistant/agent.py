
"""
Personal Knowledge Assistant.

Module 1: Google Agent Development Kit
Purpose: Build a conversational single-agent application
using Google's Gemini model.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent


# ---------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------

# Resolve the directory containing this Python file.
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the local .env file.
# Existing system environment variables are not overwritten.
load_dotenv(BASE_DIR / ".env", override=False)


def configure_gemini_credentials() -> None:
    """
    Configure Google Gemini authentication.

    Maps our application-specific GEMINI_API_KEY
    to the GOOGLE_API_KEY expected by Google GenAI.
    """

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Allow an existing GOOGLE_API_KEY as an alternative.
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key and gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Gemini API key missing. "
            "Configure GEMINI_API_KEY in your .env file."
        )


# ---------------------------------------------------------
# 2. APPLICATION CONFIGURATION
# ---------------------------------------------------------

configure_gemini_credentials()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


# ---------------------------------------------------------
# 3. AGENT INSTRUCTIONS
# ---------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are a professional Personal Knowledge Assistant.

Your primary responsibilities are:

1. Explain technical and academic concepts clearly.
2. Help users understand Python and AI engineering.
3. Provide structured, logically organized answers.
4. Explain complex topics using practical examples.
5. Distinguish verified facts from assumptions.
6. Never fabricate information when uncertain.
7. Ask clarifying questions when a request is ambiguous.

Communication guidelines:

- Be technically accurate.
- Explain the reasoning behind recommendations.
- Prefer practical examples over unnecessary jargon.
- Use readable formatting.
- Avoid unnecessarily long responses.
- Never claim to have executed code or accessed
  external systems unless that actually happened.

You are currently operating as a single conversational
agent without external tools.
"""


# ---------------------------------------------------------
# 4. ROOT AGENT DEFINITION
# ---------------------------------------------------------

root_agent = LlmAgent(
    name="personal_knowledge_assistant",
    model=MODEL_NAME,
    description=(
        "A conversational AI assistant for technical "
        "learning and knowledge exploration."
    ),
    instruction=SYSTEM_INSTRUCTION,
)
