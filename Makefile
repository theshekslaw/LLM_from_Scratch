.PHONY: help check-hardware setup download-data run train generate clean

help: ## List available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

check-hardware: ## Detect compute platform (NVIDIA GPU, Apple Silicon, or CPU)
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "Detected: NVIDIA GPU (CUDA)"; \
	elif [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "Detected: Apple Silicon (MPS)"; \
	else \
		echo "Detected: CPU-only"; \
	fi

setup: check-hardware ## Install PyTorch (correct backend) and project dependencies
	@echo "Checking prerequisites..."
	@python3 --version | grep -q "3.1[2-9]\|3.[2-9][0-9]" || (echo "Error: Python 3.12+ is required" && exit 1)
	@command -v uv > /dev/null 2>&1 || (echo "Error: uv is not installed. See https://docs.astral.sh/uv/" && exit 1)
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "Installing PyTorch for CUDA..."; \
		uv pip install torch --index-url https://download.pytorch.org/whl/cu124; \
	elif [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "Installing PyTorch for Apple Silicon (MPS)..."; \
		uv pip install torch; \
	else \
		echo "Installing PyTorch for CPU..."; \
		uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
	fi
	@echo "Syncing project dependencies..."
	uv sync
	@echo "Setup complete!"

download-data: ## Download training dataset
	python -m utils.dataset

run: ## Run the main script
	python main.py

train: ## Train the model (pass ARGS="--epochs 20" for custom args)
	python train.py $(ARGS)

generate: ## Generate text from a checkpoint (pass ARGS="--checkpoint out/model.pt")
	python -c "import torch; from tokenizer import BPETokenizer; from transformer import GPTModel, generate; \
		ckpt = torch.load('out/model.pt', weights_only=False); \
		model = GPTModel(ckpt['config']); model.load_state_dict(ckpt['model']); model.eval(); \
		tok = BPETokenizer('gpt2'); \
		ids = torch.tensor([tok.encode('Every effort moves you')]); \
		out = generate(model, ids, max_new_tokens=100, temperature=0.8, top_k=40); \
		print(tok.decode(out[0].tolist()))"

clean: ## Remove __pycache__, .venv, and other generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv out/
	@echo "Cleaned up generated files."
