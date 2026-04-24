from app.config import settings


def get_openai_client():
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package is missing. Install requirements.txt first.") from exc

    return OpenAI(api_key=settings.openai_api_key)
