import json

from app.config import settings
from app.services.openai_service import get_openai_client


def generate_plain_answer(message: str) -> dict[str, str | float]:
    prompt = (
        "You are a customer support assistant. Reply briefly and helpfully to the "
        "customer message based only on your own reasoning, without external context. "
        "Return valid JSON with keys: text, confidence_percent, confidence_basis. "
        "confidence_percent must be a number from 0 to 100."
    )
    raw_content = _generate_completion(prompt=prompt, user_message=message)

    try:
        parsed = json.loads(raw_content)
        text = str(parsed["text"]).strip()
        confidence_percent = float(parsed["confidence_percent"])
        confidence_basis = str(parsed["confidence_basis"]).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "text": raw_content,
            "confidence_percent": 55.0,
            "confidence_basis": "Fallback confidence because structured parsing failed.",
        }

    return {
        "text": text,
        "confidence_percent": round(min(100.0, max(0.0, confidence_percent)), 1),
        "confidence_basis": confidence_basis or "Model self-reported confidence.",
    }


def generate_rag_grounded_answer(message: str, retrieved_context: str) -> str:
    prompt = (
        "You are a customer support assistant. Use the retrieved support examples "
        "to answer the user's message. Be concise, actionable, and grounded in the "
        "retrieved context. The retrieval quality has already been checked before this step, "
        "so focus on giving the best grounded answer you can from the provided examples. "
        "If the examples do not fully solve the issue, say what they suggest and be transparent "
        "about the limit.\n\nRetrieved context:\n"
        f"{retrieved_context}"
    )
    return _generate_completion(prompt=prompt, user_message=message)


def predict_priority_zero_shot(message: str) -> dict[str, str | float]:
    prompt = (
        "You are classifying customer-support ticket priority. Return valid JSON with keys: "
        "priority, confidence_percent, confidence_basis. The priority must be one of "
        "urgent or normal. "
        "confidence_percent must be a number from 0 to 100."
    )
    raw_content = _generate_completion(prompt=prompt, user_message=message)

    try:
        parsed = json.loads(raw_content)
        priority = str(parsed["priority"]).strip().lower()
        confidence_percent = float(parsed["confidence_percent"])
        confidence_basis = str(parsed["confidence_basis"]).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "priority": "normal",
            "confidence_percent": 55.0,
            "confidence_basis": "Fallback confidence because structured parsing failed.",
        }

    if priority not in {"urgent", "normal"}:
        priority = "normal"

    return {
        "priority": priority,
        "confidence_percent": round(min(100.0, max(0.0, confidence_percent)), 1),
        "confidence_basis": confidence_basis or "Model self-reported confidence.",
    }


def _generate_completion(prompt: str, user_message: str) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI returned empty content.")
    return content.strip()
