from fastapi import FastAPI, HTTPException
import psutil

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            "status": "healthy",
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
