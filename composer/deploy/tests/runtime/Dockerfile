ARG FACADE_IMAGE
ARG WRAPPER_IMAGE
FROM ${WRAPPER_IMAGE} AS recordings
FROM ${FACADE_IMAGE}
RUN pip install --no-cache-dir langchain-nvidia-ai-endpoints==1.2.1 && pip check
COPY --from=recordings /app/course_cache /recordings
COPY smoke.py /app/smoke.py
ENTRYPOINT ["python", "/app/smoke.py"]
