from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from .utils import env_value, load_known_env_files


def get_openai_client():
    from openai import OpenAI
    api_key = env_value("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY")
    if not api_key:
        load_known_env_files()
        api_key = env_value("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put OPENAI_API_KEY=... in Drive .env, then rerun 00_setup_and_data.ipynb.")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError("OpenAI response did not include output_text")
    return text


def call_openai_json(client: Any, model: str, system: str, user: str, schema: Dict[str, Any], max_retries: int = 3, sleep: float = 2.0) -> Dict[str, Any]:
    """Call OpenAI Responses API with JSON schema output.

    The code intentionally uses the Responses API because the project uses structured outputs.
    """
    name = schema.get("title") or schema.get("name") or "lexnorm_output"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            )
            text = _extract_text(response).strip()
            return json.loads(text)
        except Exception as exc:
            last_error = exc
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"OpenAI JSON call failed after {max_retries} attempts: {last_error}")
