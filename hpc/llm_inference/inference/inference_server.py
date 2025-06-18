#!/usr/bin/env python3
# inference_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="LLM Inference Server")

# Load model globally with optimized settings
llm = LLM(
    model="microsoft/phi-4",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    dtype="bfloat16"  # Use BF16 for RTX 8000
)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class InferenceResponse(BaseModel):
    generated_text: str
    prompt: str

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return InferenceResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)