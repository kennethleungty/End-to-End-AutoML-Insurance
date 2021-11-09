from fastapi import FastAPI

# Create API instance
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}