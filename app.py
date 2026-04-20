

from server import app  # re-export for ASGI servers

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
