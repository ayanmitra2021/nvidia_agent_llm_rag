"""Shared OpenAI request compatibility without provider configuration."""
from copy import deepcopy
import math


def numeric_json(value):
    """Normalize only equivalent JSON numbers for lookup, never stored evidence."""
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number in model request")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: numeric_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numeric_json(item) for item in value]
    return value  # Booleans and number-looking strings remain distinct.


def structured_output(body):
    """Translate NVIDIA SDK guidance to portable OpenAI JSON-schema controls."""
    result = deepcopy(body)
    nvext = result.get('nvext')
    if nvext is not None and not isinstance(nvext, dict):
        raise ValueError('nvext must be an object')
    formats = [result['response_format']] if result.get('response_format') is not None else []
    legacy = False
    for container in (result, nvext or {}):
        if 'guided_json' in container:
            schema = container.pop('guided_json')
            if not isinstance(schema, (dict, bool)):
                raise ValueError('guided_json must be a JSON schema')
            formats.append({'type': 'json_schema', 'json_schema': {
                'name': 'response', 'schema': schema, 'strict': True}})
            legacy = True
    if len(formats) > 1:
        raise ValueError('Use one structured-output format per request')
    if legacy:
        if result.get('nvext') == {}:
            result.pop('nvext')
        result['response_format'] = formats[0]
    return result


def completion_limit(body):
    """Forward one modern token limit, with explicit modern values taking precedence.

    ChatNVIDIA can still serialize its default max_tokens alongside a bound
    max_completion_tokens. Preserve the requested modern limit in that case.
    """
    result = dict(body)
    limit = result.pop("max_tokens", None)
    modern = result.pop("max_completion_tokens", None)
    if modern is not None:
        limit = modern
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Completion token limit must be a positive integer")
        result["max_completion_tokens"] = limit
    return result


def facade_chat_request(body, default_thinking_model="nvidia/nemotron-3.5-lightning-30b-a3b"):
    """Apply the facade's explicit chat defaults without changing caller input."""
    result = structured_output(completion_limit(body))
    if result.get("stop") == []:
        result.pop("stop")
    if result.get("model") in {default_thinking_model, "nvidia/nemotron-3-super-120b-a12b"}:
        result.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return result
