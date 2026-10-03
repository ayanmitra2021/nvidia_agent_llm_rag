"""Explicit provider dialect settings; no endpoint or credential catalog."""
import copy
import os
import re


def env_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError("invalid environment reference")
    return value


def provider_url(entry):
    if ("base_url" in entry) == ("base_url_env" in entry):
        raise ValueError("supply exactly one provider URL or environment reference")
    if "base_url_env" in entry:
        value = os.getenv(env_name(entry["base_url_env"]))
        if not value:
            raise ValueError("missing provider URL environment variable")
        return value
    return entry["base_url"]


def compatibility(entry):
    budget = entry.get("reasoning_budget_parameter", "reasoning_budget")
    field = entry.get("reasoning_response_field", "reasoning_content")
    if budget not in {"reasoning_budget", "thinking_token_budget"}:
        raise ValueError("unsupported reasoning budget parameter")
    if field not in {"reasoning", "reasoning_content"}:
        raise ValueError("unsupported reasoning response field")
    return {"reasoning_budget_parameter": budget, "reasoning_response_field": field}


def request_body(body, settings):
    body = copy.deepcopy(body)
    target = settings["reasoning_budget_parameter"]
    if target != "reasoning_budget" and "reasoning_budget" in body:
        if target in body and body[target] != body["reasoning_budget"]:
            raise ValueError("conflicting reasoning budgets")
        body[target] = body.pop("reasoning_budget")
    return body


def response_body(body, settings):
    """Preserve provider fields and expose the stable course reasoning field."""
    body = copy.deepcopy(body)
    field = settings["reasoning_response_field"]
    if field == "reasoning_content":
        return body
    for choice in body.get("choices", []):
        for name in ("message", "delta"):
            message = choice.get(name)
            if not isinstance(message, dict) or message.get("reasoning_content"):
                continue
            reasoning = message.get(field)
            if reasoning is None:
                reasoning = (message.get("provider_specific_fields") or {}).get(field)
            if isinstance(reasoning, str):
                message["reasoning_content"] = reasoning
    return body
