FROM python:3.11-slim
WORKDIR /app
COPY llm_client-requirements.txt litellm_wrapper-requirements.txt ./
RUN pip install --no-cache-dir -r litellm_wrapper-requirements.txt && pip check
COPY litellm_wrapper.py litellm_service.py llm_output.py llm_store.py llm_cache_bundle.py llm_course_cache.py llm_cache_pacing.py ./
COPY provider_compat.py provider_policy.py llm_ttft.py llm_circuit.py llm_request.py llm_tracing.py llm_profile.py llm_runtime.py ./
COPY course_cache/ ./course_cache/
RUN PYTHONDONTWRITEBYTECODE=1 python -c "from llm_course_cache import CourseCache; cache = CourseCache(); cache._load(); assert cache.entries and not cache.unavailable, 'Required course cache unavailable'"
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9001/health')"
ENTRYPOINT ["python", "litellm_wrapper.py"]
