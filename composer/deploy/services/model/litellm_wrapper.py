"""Private credential owner. Start through this entry point in every environment."""
from llm_output import protect_output, diagnostic

protect_output()
import os
os.umask(0o077)
if not os.getenv("NEBIUS_TOKEN_FACTORY_KEY") and os.getenv("NEBUIS_TOKEN_FACTORY_KEY"):
    os.environ["NEBIUS_TOKEN_FACTORY_KEY"] = os.environ["NEBUIS_TOKEN_FACTORY_KEY"]
# Enable the registered fallback when its credential is supplied by /dli/secrets.
# An explicit deployment URL still takes precedence.
if os.getenv("NEBIUS_TOKEN_FACTORY_KEY"):
    os.environ.setdefault("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

if __name__ == "__main__":
    try:
        import uvicorn
        from litellm_service import app
        uvicorn.run(app, host="0.0.0.0", port=9001, access_log=False, log_config=None)
    except Exception:
        diagnostic("configuration_error")
        raise SystemExit(1) from None
