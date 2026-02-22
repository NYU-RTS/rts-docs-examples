# Srart llama.cpp server in the terminal
apptainer exec --nv llama.cpp_server-cuda.sif bash -lc '
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH
/app/llama-server \
  -m /scratch/$USER/models/qwen2.5-0.5b-gguf-fp16/Qwen2.5-0.5B-Instruct.FP16.gguf \
  --host 127.0.0.1 --port 8000 \
  -ngl 999 \
  -c 4096 \
  -np 8 \
  -b 512 \
  -ub 512
'

# after the server is running, open a second terminal
python -m guidellm benchmark run \
  --target "http://127.0.0.1:8000" \
  --backend-type openai_http \
  --model "Qwen2.5-0.5B-Instruct.FP16.gguf" \
  --processor "Qwen/Qwen2.5-0.5B-Instruct" \
  --processor-args '{"trust_remote_code": true}' \
  --data "prompt_tokens=256,output_tokens=256" \
  --rate-type concurrent \
  --rate 8 \
  --max-requests 64 \
  --max-seconds 600 \
  --random-seed 0 \
  --output-path ./llama_server_cuda_qwen05b_fp16_rate8.json
