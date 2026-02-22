# Start the vLLM server in the terminal

apptainer exec --nv /scratch/NET_ID/vllm_test/vllm-openai_latest.sif \
  vllm serve "Qwen/Qwen2.5-0.5B-Instruct" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16

# after the server is running, open a second terminal
python -m guidellm benchmark run \
  --target "http://127.0.0.1:8000" \
  --backend-type openai_http \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --processor "Qwen/Qwen2.5-0.5B-Instruct" \
  --processor-args '{"trust_remote_code": true}' \
  --data "prompt_tokens=256,output_tokens=256" \
  --rate-type concurrent \
  --rate 8 \
  --max-requests 64 \
  --random-seed 0 \
  --output-path ./vllm_fp16_concurrent8.json
