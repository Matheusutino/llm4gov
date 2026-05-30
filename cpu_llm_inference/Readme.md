# Inferência em CPU no LLM4Gov

O **LLM4Gov** propõe o uso de modelos de linguagem em ambientes governamentais com foco em **privacidade**, **baixo custo**, **controle institucional** e **soberania tecnológica**.

A inferência em CPU é uma estratégia importante nesse contexto porque permite executar modelos localmente, usando servidores já disponíveis nas instituições. Isso reduz a dependência de GPUs, diminui o custo inicial de implantação e facilita a criação de pilotos e aplicações internas.

Em órgãos públicos, muitos dados não devem ser enviados para serviços externos. Documentos administrativos, processos, registros acadêmicos, informações institucionais e dados sensíveis precisam permanecer em ambientes controlados. Ao executar modelos localmente, o LLM4Gov reduz riscos de exposição e facilita o cumprimento de políticas de segurança e proteção de dados.

A execução local também fortalece a **soberania tecnológica**. A instituição passa a controlar o modelo utilizado, a infraestrutura, os logs, as versões, as atualizações e os mecanismos de auditoria. Isso evita dependência excessiva de plataformas proprietárias e permite maior transparência sobre como a solução funciona.

Do ponto de vista operacional, a inferência em CPU permite uma adoção gradual. É possível começar com modelos compactos e quantizados, ajustar parâmetros de desempenho, criar múltiplas instâncias locais e distribuir requisições conforme a capacidade do servidor. Essa abordagem torna o uso de modelos de linguagem mais próximo da realidade de muitas instituições públicas.

O LLM4Gov não parte da premissa de que toda aplicação precisa de uma infraestrutura cara ou de modelos executados em nuvem. Muitas tarefas relevantes, como classificação de textos, extração de informações, sumarização, busca semântica e apoio à análise documental, podem ser executadas em ambiente local com desempenho adequado.

Assim, a inferência em CPU representa uma escolha prática para ampliar o uso responsável de IA no setor público. Ela combina menor custo, maior privacidade, controle institucional e independência tecnológica, alinhando o desenvolvimento de soluções de IA às restrições reais de infraestrutura, orçamento e governança das instituições.

A seguir, apresentamos um tutorial base utilizado nos projetos do **LLM4Gov** para configurar inferência local em CPU. O tutorial mostra como compilar o `llama.cpp`, baixar modelos em formato GGUF, configurar múltiplas instâncias locais, ajustar parâmetros de contexto e threads, criar um serviço de inicialização automática e testar o desempenho em paralelo. Essa configuração serve como ponto de partida para implantações em ambientes institucionais, podendo ser adaptada de acordo com a CPU, a memória disponível, o modelo escolhido e o volume esperado de requisições.


# Running a Local Gemma GGUF Pool with llama.cpp on CPU

This tutorial describes how to install, compile, configure, and run `llama.cpp` on a Linux CPU server using multiple `llama-server` instances.

The setup below was designed for a CPU-only machine with:

```text
CPU: Intel Xeon E5-2640
Logical CPUs: 24
RAM: 96 GB
NUMA nodes: 2
Model: Gemma 4 E2B GGUF
Runtime: llama.cpp
```

The same procedure can be adapted to other CPUs. The most important parameters to tune are:

```text
NUM_SERVERS
CTX_SIZE
THREADS_PER_SERVER
THREADS_BATCH_PER_SERVER
BATCH_SIZE
UBATCH_SIZE
```

---

## 1. Important idea

There are two common ways to serve multiple requests with `llama.cpp`.

One option is a single `llama-server` process using `--parallel`.

The other option is to run several independent `llama-server` processes, each one on a different port.

In this setup, we use the second approach:

```text
one systemd service
one configuration file
one startup script
multiple llama-server processes
one port per process
```

Example:

```text
8081 -> llama-server instance 1
8082 -> llama-server instance 2
8083 -> llama-server instance 3
...
```

This approach is simple to monitor and easy to tune.

---

## 2. Directory layout

The setup uses these directories:

```text
/opt/llama.cpp
```

Source code and compiled binaries of `llama.cpp`.

```text
/opt/llama.cpp/build-cpu/bin/llama-server
```

Compiled `llama-server` binary.

```text
/opt/llama.cpp/build-cpu/bin/llama-cli
```

Compiled `llama-cli` binary.

```text
/models/hf-cache
```

Hugging Face model cache.

```text
/opt/llama-services
```

Configuration files, scripts, and test tools.

```text
/var/log/llama
```

Logs for each `llama-server` instance.

---

## 3. Install dependencies

Run as root:

```bash
apt-get update

apt-get install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  ccache \
  pkg-config \
  libopenblas-dev \
  libgomp1 \
  numactl \
  curl \
  python3
```

---

## 4. Clone llama.cpp

```bash
cd /opt

git clone https://github.com/ggml-org/llama.cpp

cd /opt/llama.cpp
```

---

## 5. Compile llama.cpp for CPU

This build is optimized for CPU inference.

```bash
cd /opt/llama.cpp

rm -rf build-cpu

cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_LTO=ON \
  -DGGML_CCACHE=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-cpu --config Release -j 24
```

Check the binaries:

```bash
ls -lh /opt/llama.cpp/build-cpu/bin/llama-server
ls -lh /opt/llama.cpp/build-cpu/bin/llama-cli
ls -lh /opt/llama.cpp/build-cpu/bin/llama-bench
```

---

## 6. Notes about CPU-specific optimization

The option:

```bash
-DGGML_NATIVE=ON
```

compiles `llama.cpp` for the CPU of the current machine.

This is good for performance, but the compiled binary may not be ideal for another machine with a different CPU.

Before compiling, check your CPU:

```bash
lscpu
```

Look at:

```text
CPU model
number of sockets
number of cores
threads per core
NUMA nodes
CPU flags
```

For example, on an older Intel Xeon E5-2640, the CPU supports AVX but not AVX2 or AVX512. In this case, compiling directly on the target machine with `GGML_NATIVE=ON` is the safest choice.

For newer Intel CPUs, especially those with AVX2, AVX512, VNNI, or AMX, performance can be much better. The best compilation flags may be different.

For AMD CPUs, also compile directly on the target machine and benchmark different thread counts.

The general rule is:

```text
compile on the same machine where the model will run
benchmark different thread values
do not assume that using all CPU threads is always faster
```

---

## 7. Hugging Face cache directory

Create the cache directory:

```bash
mkdir -p /models/hf-cache
mkdir -p /models/hf-cache/hub
```

The environment variables used in this setup are:

```bash
HF_HOME=/models/hf-cache
HF_HUB_CACHE=/models/hf-cache/hub
```

This keeps downloaded GGUF files outside `/root/.cache`.

---

## 8. Test a single model call

Before creating the service, test the model manually.

```bash
MODEL_HF="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"

export HF_HOME=/models/hf-cache
export HF_HUB_CACHE=/models/hf-cache/hub
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

/opt/llama.cpp/build-cpu/bin/llama-cli \
  -hf "$MODEL_HF" \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  -p "Explain text mining in one short paragraph." \
  -n 256 \
  -c 10240 \
  -t 4 \
  -tb 12 \
  -b 2048 \
  -ub 512 \
  --flash-attn auto \
  --mlock
```

If this works, the model is downloaded and cached.

---

## 9. Parameter meaning

The most important runtime parameters are:

```bash
-c
```

Context size for each server instance.

Example:

```bash
-c 10240
```

means each instance can use a context window of 10240 tokens.

```bash
-t
```

Number of CPU threads used mainly during token generation.

```bash
-tb
```

Number of CPU threads used during batch and prompt processing.

If prompts are large, increasing `-tb` can help. However, if many servers run at the same time, a high `-tb` can overload the CPU.

```bash
-b
```

Logical batch size.

```bash
-ub
```

Physical micro-batch size.

```bash
--reasoning off
--reasoning-budget 0
```

Disable thinking/reasoning mode.

```bash
--mlock
```

Try to keep model memory locked in RAM.

```bash
--no-mmap
```

Load the model more directly into process memory instead of relying on memory mapping.

```bash
--metrics
```

Expose metrics endpoints.

---

## 10. Create service directories

```bash
mkdir -p /opt/llama-services
mkdir -p /var/log/llama
mkdir -p /models/hf-cache
mkdir -p /models/hf-cache/hub
```

---

## 11. Create the configuration file

Create:

```bash
nano /opt/llama-services/gemma-pool.conf
```

Content:

```bash
# Model
MODEL_HF="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"

# llama.cpp binary
LLAMA_BIN="/opt/llama.cpp/build-cpu/bin/llama-server"

# Hugging Face cache
HF_HOME="/models/hf-cache"
HF_HUB_CACHE="/models/hf-cache/hub"

# Network
HOST="0.0.0.0"
BASE_PORT=8081

# Number of independent llama-server processes
NUM_SERVERS=6

# Context size per server
CTX_SIZE=30720

# CPU threads per server
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2

# Batch settings
BATCH_SIZE=2048
UBATCH_SIZE=512

# Disable thinking/reasoning
REASONING="off"
REASONING_BUDGET=0

# Memory options
USE_MLOCK=1
USE_NO_MMAP=1
```

To change the setup later, edit only this file.

Examples:

```bash
NUM_SERVERS=10
CTX_SIZE=10240
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
```

or:

```bash
NUM_SERVERS=6
CTX_SIZE=30720
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=4
```

After editing, restart the service:

```bash
systemctl restart llama-gemma-pool
```

---

## 12. Create the startup script

Create:

```bash
nano /opt/llama-services/start-gemma-pool.sh
```

Content:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /opt/llama-services/gemma-pool.conf

export HF_HOME
export HF_HUB_CACHE
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${THREADS_PER_SERVER}"

declare -a PIDS=()

if [ ! -x "${LLAMA_BIN}" ]; then
  echo "ERROR: llama-server not found or not executable: ${LLAMA_BIN}"
  exit 1
fi

start_instance() {
  local idx="$1"
  local port="$2"

  local extra_flags=()

  if [ "${USE_MLOCK}" = "1" ]; then
    extra_flags+=(--mlock)
  fi

  if [ "${USE_NO_MMAP}" = "1" ]; then
    extra_flags+=(--no-mmap)
  fi

  echo "Starting instance $((idx + 1))/${NUM_SERVERS}: port=${port}, ctx=${CTX_SIZE}, threads=${THREADS_PER_SERVER}"

  "${LLAMA_BIN}" \
    -hf "${MODEL_HF}" \
    --host "${HOST}" \
    --port "${port}" \
    --parallel 1 \
    -c "${CTX_SIZE}" \
    -t "${THREADS_PER_SERVER}" \
    -tb "${THREADS_BATCH_PER_SERVER}" \
    -b "${BATCH_SIZE}" \
    -ub "${UBATCH_SIZE}" \
    --jinja \
    --reasoning "${REASONING}" \
    --reasoning-budget "${REASONING_BUDGET}" \
    --flash-attn auto \
    --metrics \
    "${extra_flags[@]}" \
    > "/var/log/llama/gemma-${port}.log" \
    2> "/var/log/llama/gemma-${port}.err" &

  PIDS+=("$!")
}

stop_all() {
  echo "Stopping all llama-server instances..."

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  wait || true
}

trap stop_all SIGTERM SIGINT EXIT

for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  start_instance "$i" "$port"
done

echo "All ${NUM_SERVERS} instances started."

wait -n

echo "One instance exited. Stopping the pool..."
stop_all
exit 1
```

Make it executable:

```bash
chmod +x /opt/llama-services/start-gemma-pool.sh
```

---

## 13. Create the systemd service

Create:

```bash
nano /etc/systemd/system/llama-gemma-pool.service
```

Content:

```ini
[Unit]
Description=llama.cpp Gemma pool
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/llama.cpp

ExecStart=/opt/llama-services/start-gemma-pool.sh

Restart=always
RestartSec=10

KillMode=control-group
TimeoutStopSec=60

LimitNOFILE=1048576
LimitMEMLOCK=infinity

StandardOutput=append:/var/log/llama/gemma-pool.log
StandardError=append:/var/log/llama/gemma-pool.err

[Install]
WantedBy=multi-user.target
```

Reload systemd:

```bash
systemctl daemon-reload
```

---

## 14. Start the service

```bash
systemctl start llama-gemma-pool
```

Check status:

```bash
systemctl status llama-gemma-pool
```

Follow logs:

```bash
journalctl -u llama-gemma-pool -f
```

Check the pool log:

```bash
tail -f /var/log/llama/gemma-pool.log
```

Check one instance log:

```bash
tail -f /var/log/llama/gemma-8081.log
```

Check one instance error log:

```bash
tail -f /var/log/llama/gemma-8081.err
```

---

## 15. Enable startup at boot

To make the service start automatically when the machine boots:

```bash
systemctl enable llama-gemma-pool
```

To disable automatic startup:

```bash
systemctl disable llama-gemma-pool
```

To restart manually:

```bash
systemctl restart llama-gemma-pool
```

To stop manually:

```bash
systemctl stop llama-gemma-pool
```

---

## 16. Check running ports

If `BASE_PORT=8081` and `NUM_SERVERS=6`, the active ports should be:

```text
8081
8082
8083
8084
8085
8086
```

Check:

```bash
ss -ltnp | grep llama-server
```

Or:

```bash
for port in $(seq 8081 8086); do
  echo "===== $port ====="
  curl -s http://localhost:$port/props | head -c 300
  echo
done
```

If you use `NUM_SERVERS=10`, check ports `8081` to `8090`.

---

## 17. Test one server

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-e2b",
    "messages": [
      {"role": "user", "content": "Explain text mining in one short sentence."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

---

## 18. Create a parallel test script

Create:

```bash
nano /opt/llama-services/test_llama_pool.py
```

Content:

```python
#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def call_server(port, request_id, args):
    url = f"http://{args.host}:{port}/v1/chat/completions"

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.prompt.format(id=request_id, port=port)
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.time()

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read()
            status = resp.status

        end = time.time()
        elapsed = end - start

        parsed = json.loads(body.decode("utf-8", errors="replace"))

        usage = parsed.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        content = ""
        try:
            content = parsed["choices"][0]["message"]["content"]
        except Exception:
            content = ""

        return {
            "ok": True,
            "port": port,
            "request_id": request_id,
            "status": status,
            "seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "chars": len(content),
            "error": None,
        }

    except urllib.error.HTTPError as e:
        end = time.time()
        return {
            "ok": False,
            "port": port,
            "request_id": request_id,
            "status": e.code,
            "seconds": end - start,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "chars": 0,
            "error": f"HTTPError: {e}",
        }

    except Exception as e:
        end = time.time()
        return {
            "ok": False,
            "port": port,
            "request_id": request_id,
            "status": None,
            "seconds": end - start,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "chars": 0,
            "error": repr(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel test for multiple llama-server instances."
    )

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--end-port", type=int, required=True)

    parser.add_argument("--requests-per-server", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)

    parser.add_argument("--model", default="gemma-e2b")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=900)

    parser.add_argument(
        "--prompt",
        default=(
            "Request {id}, server port {port}. "
            "Explain in about 100 words what text mining is."
        )
    )

    parser.add_argument("--json-out", default=None)

    args = parser.parse_args()

    ports = list(range(args.start_port, args.end_port + 1))
    jobs = []

    request_id = 1
    for _ in range(args.requests_per_server):
        for port in ports:
            jobs.append((port, request_id))
            request_id += 1

    max_workers = args.max_workers or len(jobs)

    print("=" * 70)
    print("LLAMA POOL PARALLEL TEST")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Ports: {args.start_port}-{args.end_port}")
    print(f"Servers: {len(ports)}")
    print(f"Requests per server: {args.requests_per_server}")
    print(f"Total requests: {len(jobs)}")
    print(f"Max workers: {max_workers}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 70)

    start_all = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(call_server, port, req_id, args)
            for port, req_id in jobs
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = "OK" if result["ok"] else "ERR"
            print(
                f"[{status}] "
                f"req={result['request_id']} "
                f"port={result['port']} "
                f"status={result['status']} "
                f"time={result['seconds']:.2f}s "
                f"tokens={result['total_tokens']} "
                f"chars={result['chars']} "
                f"error={result['error']}"
            )

    end_all = time.time()
    wall_time = end_all - start_all

    ok_results = [r for r in results if r["ok"]]
    error_results = [r for r in results if not r["ok"]]

    latencies = [r["seconds"] for r in ok_results]

    total_prompt_tokens = sum(
        r["prompt_tokens"] for r in ok_results
        if isinstance(r["prompt_tokens"], int)
    )

    total_completion_tokens = sum(
        r["completion_tokens"] for r in ok_results
        if isinstance(r["completion_tokens"], int)
    )

    total_tokens = sum(
        r["total_tokens"] for r in ok_results
        if isinstance(r["total_tokens"], int)
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total wall time: {wall_time:.2f}s")
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {len(ok_results)}")
    print(f"Failed requests: {len(error_results)}")
    print(f"Requests/s: {len(ok_results) / wall_time:.3f}")

    if latencies:
        print()
        print("Latency, seconds:")
        print(f"  min:  {min(latencies):.2f}")
        print(f"  avg:  {statistics.mean(latencies):.2f}")
        print(f"  p50:  {percentile(latencies, 50):.2f}")
        print(f"  p90:  {percentile(latencies, 90):.2f}")
        print(f"  p95:  {percentile(latencies, 95):.2f}")
        print(f"  max:  {max(latencies):.2f}")

    print()
    print("Tokens:")
    print(f"  prompt tokens:     {total_prompt_tokens}")
    print(f"  completion tokens: {total_completion_tokens}")
    print(f"  total tokens:      {total_tokens}")

    if total_tokens > 0:
        print(f"  total tokens/s:      {total_tokens / wall_time:.2f}")
    if total_completion_tokens > 0:
        print(f"  completion tokens/s: {total_completion_tokens / wall_time:.2f}")

    print()
    print("Per-port summary:")
    for port in ports:
        port_results = [r for r in results if r["port"] == port]
        port_ok = [r for r in port_results if r["ok"]]
        port_lat = [r["seconds"] for r in port_ok]

        if port_lat:
            print(
                f"  {port}: "
                f"ok={len(port_ok)}/{len(port_results)} "
                f"avg={statistics.mean(port_lat):.2f}s "
                f"max={max(port_lat):.2f}s"
            )
        else:
            print(f"  {port}: ok=0/{len(port_results)}")

    if args.json_out:
        output = {
            "summary": {
                "host": args.host,
                "start_port": args.start_port,
                "end_port": args.end_port,
                "servers": len(ports),
                "requests_per_server": args.requests_per_server,
                "total_requests": len(results),
                "successful_requests": len(ok_results),
                "failed_requests": len(error_results),
                "wall_time_seconds": wall_time,
                "requests_per_second": len(ok_results) / wall_time if wall_time > 0 else None,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_tokens_per_second": total_tokens / wall_time if wall_time > 0 and total_tokens > 0 else None,
                "completion_tokens_per_second": total_completion_tokens / wall_time if wall_time > 0 and total_completion_tokens > 0 else None,
            },
            "results": results,
        }

        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print()
        print(f"JSON saved to: {args.json_out}")


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x /opt/llama-services/test_llama_pool.py
```

---

## 19. Run a parallel test

For 6 servers on ports `8081` to `8086`:

```bash
python3 /opt/llama-services/test_llama_pool.py \
  --start-port 8081 \
  --end-port 8086 \
  --requests-per-server 1 \
  --max-tokens 192
```

For a heavier test:

```bash
python3 /opt/llama-services/test_llama_pool.py \
  --start-port 8081 \
  --end-port 8086 \
  --requests-per-server 3 \
  --max-tokens 192
```

Save results to JSON:

```bash
python3 /opt/llama-services/test_llama_pool.py \
  --start-port 8081 \
  --end-port 8086 \
  --requests-per-server 3 \
  --max-tokens 192 \
  --json-out /tmp/llama_pool_test.json
```

---

## 20. Monitor CPU and RAM

```bash
free -h
```

```bash
ps -C llama-server -o pid,psr,pcpu,pmem,rss,vsz,cmd --sort=pid
```

Total RAM used by all `llama-server` processes:

```bash
ps -C llama-server -o rss= | awk '{sum+=$1} END {printf "Total RSS llama-server: %.2f GB\n", sum/1024/1024}'
```

Live monitoring:

```bash
watch -n 1 'free -h; echo; ps -C llama-server -o pid,psr,pcpu,pmem,rss,cmd --sort=pid'
```

---

## 21. How to tune the setup

Edit:

```bash
nano /opt/llama-services/gemma-pool.conf
```

Then restart:

```bash
systemctl restart llama-gemma-pool
```

### More servers, smaller context

```bash
NUM_SERVERS=10
CTX_SIZE=10240
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
```

Good when many small/medium requests arrive at the same time.

### Fewer servers, larger context

```bash
NUM_SERVERS=6
CTX_SIZE=30720
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
```

Good when each request needs a large context.

### More prompt-processing threads

```bash
NUM_SERVERS=6
CTX_SIZE=30720
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=4
```

This may help when prompts are large, but it can overload the CPU if many requests arrive at the same time.

### Generation vs prompt threads

`THREADS_PER_SERVER` maps to:

```bash
-t
```

This affects token generation.

`THREADS_BATCH_PER_SERVER` maps to:

```bash
-tb
```

This affects prompt and batch processing.

For many independent servers, start with:

```bash
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
```

Then benchmark.

---

## 22. Recommended initial configurations

For this CPU-only server, a conservative configuration is:

```bash
NUM_SERVERS=6
CTX_SIZE=30720
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
BATCH_SIZE=2048
UBATCH_SIZE=512
```

A higher-concurrency configuration is:

```bash
NUM_SERVERS=10
CTX_SIZE=10240
THREADS_PER_SERVER=2
THREADS_BATCH_PER_SERVER=2
BATCH_SIZE=2048
UBATCH_SIZE=512
```

The best configuration depends on the workload.

Use the parallel test script to compare:

```bash
python3 /opt/llama-services/test_llama_pool.py \
  --start-port 8081 \
  --end-port 8086 \
  --requests-per-server 3 \
  --max-tokens 192
```

Then compare:

```text
latency average
p90 latency
p95 latency
completion tokens/s
total tokens/s
error rate
RAM usage
CPU saturation
```

---

## 23. Summary

This setup provides:

```text
CPU-only llama.cpp inference
Gemma GGUF model from Hugging Face
configurable number of servers
configurable context size
configurable threads per server
OpenAI-compatible HTTP endpoints
systemd service for automatic startup
parallel test script with basic metrics
```

Main files:

```text
/opt/llama.cpp
/opt/llama-services/gemma-pool.conf
/opt/llama-services/start-gemma-pool.sh
/opt/llama-services/test_llama_pool.py
/etc/systemd/system/llama-gemma-pool.service
/models/hf-cache
/var/log/llama
```

Start on boot:

```bash
systemctl enable llama-gemma-pool
```

Start now:

```bash
systemctl start llama-gemma-pool
```

Restart after configuration changes:

```bash
systemctl restart llama-gemma-pool
```

Stop:

```bash
systemctl stop llama-gemma-pool
```
