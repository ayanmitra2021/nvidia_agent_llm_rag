## Relevant for supplying routes
from fastapi import FastAPI

## Relevant for supporting docker utilities
import docker
from docker.errors import NotFound


client = docker.from_env()
app = FastAPI()

@app.get("/")
async def read_root():
    """Default Route: Usually good to render instructions..."""
    return {"Hello": "World"}


@app.get("/help")
async def read_root():
    """Typical help route to tell users what they can do"""
    return {"Options": "[/containers, /containers/{container_id}/logs, containers/{container_id}/restart]"}


@app.get("/healthy")
async def read_root():
    """Typical health check route. Can be used by other microservices to tell whether they can rely on this one"""
    return True


@app.get("/containers")
async def list_containers():
    """A listing route. Lists current set of containers"""
    containers = client.containers.list(all=True)
    return [{"id": container.id, "name": container.name, "status": container.status} for container in containers]


@app.get("/containers/{container_name}/logs")
async def get_container_logs(container_name: str):
    """Route that allows you to query the log file of the container"""
    try:
        container = client.containers.get(container_name)
        logs = container.logs()
        return {"logs": logs.decode('utf-8')}
    except NotFound:
        return {"error": f"Container `{container_name}` not found"}


@app.post("/containers/{container_name}/restart")
async def restart_container(container_name: str):
    """Just in case it ever becomes necessary (probably won't be though)"""
    try:
        container = client.containers.get(container_name)
        container.restart()
        return {"status": f"Container {container_name} restarted successfully"}
    except NotFound:
        return {"error": f"Container {container_name} not found"}
