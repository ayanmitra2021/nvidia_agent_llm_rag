# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get update && apt-get install -y \
    build-essential bash curl unzip wget git libgl1 tree nodejs npm && \
    rm -rf /var/lib/apt/lists/*

#Run pip dependencies
COPY composer/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip check

RUN pip install --no-cache-dir --root-user-action ignore \
    uv jupyterlab==4.5.5 jupyterlab-widgets jupyter-archive ipywidgets \
    jupyter-resource-usage jupytext jupyter-server-proxy \
    "jupyter-server>=2.0.0" jupyter-collaboration \
    jupyterlab-language-pack-zh-CN \
    jupyterlab-language-pack-ja-JP \
    jupyterlab-language-pack-zh-TW

COPY composer/jupyter/ /opt/jupyter_cfg/
RUN rm -rf /root/.jupyter/* \
    && cp /opt/jupyter_cfg/bashrc /root/.bashrc \
    && mkdir -p /root/.jupyter \
    && cp -r /opt/jupyter_cfg/.jupyter/* /root/.jupyter/ \
    && mkdir -p /usr/local/share/jupyter/lab/settings \
    && cp -r /opt/jupyter_cfg/.jupyter /root/.jupyter \
    && cp /opt/jupyter_cfg/overrides.json /usr/local/share/jupyter/lab/settings/overrides.json

WORKDIR /dli/task

# Expose port 8888 for JupyterLab
EXPOSE 8888
EXPOSE 9012

ENV SHELL=/bin/bash

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/*.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Recover text from PDFs whose embedded fonts lack Unicode mappings.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY composer/runtime/ /opt/course-runtime/
ENV PYTHONPATH=/opt/course-runtime
RUN mkdir -p /usr/local/etc/jupyter/jupyter_server_config.d
COPY composer/runtime/course-runtime.json /usr/local/etc/jupyter/jupyter_server_config.d/course-runtime.json

# Keep the exact build recipe available in the learner workspace.
COPY composer/Dockerfile /opt/course-runtime/Dockerfile
