# LLM Architecture Visualizer — Architecture Document

A web application that generates interactive Excalidraw diagrams showing the internal architecture of popular LLM models (GPT-2, LLaMA 3, Qwen3, Mistral, Gemma) with real parameter counts fetched from HuggingFace. The goal is educational — help people understand how different transformer architectures compare.

---

## 1. System Overview & Data Flow

```
User selects model (dropdown)
        │
        ▼
Frontend calls Backend API
   GET /api/architecture/{model_id}
        │
        ▼
Backend fetches config from HuggingFace
   AutoConfig.from_pretrained(model_id)
   (only config.json ~1-2KB, NOT model weights)
        │
        ▼
Backend normalizes config field names
   across model families → unified JSON
        │
        ▼
Frontend receives normalized architecture data
        │
        ▼
diagram-generator.ts converts config
   → ExcalidrawElementSkeleton[]
        │
        ▼
convertToExcalidrawElements() → full elements
        │
        ▼
excalidrawAPI.updateScene() renders
   interactive diagram on canvas
```

---

## 2. Backend (FastAPI)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | Returns curated list of supported models with display names and HuggingFace model IDs |
| `GET` | `/api/architecture/{model_id}` | Fetches config via `AutoConfig.from_pretrained()`, normalizes field names, returns unified JSON |

### Supported Models (Initial Set)

| Display Name | HuggingFace Model ID |
|---|---|
| GPT-2 | `openai-community/gpt2` |
| GPT-2 XL | `openai-community/gpt2-xl` |
| LLaMA 3 8B | `meta-llama/Meta-Llama-3-8B` |
| Qwen3 8B | `Qwen/Qwen3-8B` |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` |
| Gemma 2B | `google/gemma-2b` |

### Config Normalization

Different model families use different keys for the same concept. The backend normalizes them into a unified schema.

| Unified Field | GPT-2 key | LLaMA / Qwen / Mistral key | Gemma key |
|---|---|---|---|
| `hidden_size` | `n_embd` | `hidden_size` | `hidden_size` |
| `num_hidden_layers` | `n_layer` | `num_hidden_layers` | `num_hidden_layers` |
| `num_attention_heads` | `n_head` | `num_attention_heads` | `num_attention_heads` |
| `num_kv_heads` | `n_head` (same) | `num_key_value_heads` | `num_key_value_heads` |
| `intermediate_size` | `n_inner` or `4 * n_embd` | `intermediate_size` | `intermediate_size` |
| `vocab_size` | `vocab_size` | `vocab_size` | `vocab_size` |
| `max_position_embeddings` | `n_positions` | `max_position_embeddings` | `max_position_embeddings` |

### Derived Fields

These are computed from the normalized config, not read directly:

- **`attention_type`**: `"mha"` if `num_kv_heads == num_attention_heads`, `"gqa"` if `num_kv_heads > 1` but less than `num_attention_heads`, `"mqa"` if `num_kv_heads == 1`
- **`positional_encoding`**: `"rope"` for LLaMA/Qwen/Mistral/Gemma families, `"learned"` for GPT-2
- **`normalization`**: `"rmsnorm"` for LLaMA/Qwen/Mistral/Gemma, `"layernorm"` for GPT-2
- **`activation`**: `"silu"` / `"gelu"` based on `hidden_act` config field
- **`rope_theta`**: base frequency for RoPE (when applicable)
- **`total_params`**: estimated total parameter count (computed from dimensions)

### Response Schema

```json
{
  "model_id": "meta-llama/Meta-Llama-3-8B",
  "display_name": "LLaMA 3 8B",
  "architecture": {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_kv_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "attention_type": "gqa",
    "positional_encoding": "rope",
    "rope_theta": 500000,
    "normalization": "rmsnorm",
    "activation": "silu",
    "total_params": 8030000000
  }
}
```

### Error Handling

- Unknown model ID → `404` with message
- HuggingFace fetch failure → `502` with upstream error details
- Rate limiting → cache configs in-memory (they don't change) using `functools.lru_cache`

---

## 3. Frontend (Next.js + Excalidraw)

### Components

#### `ModelSelector.tsx`
- Dropdown populated by `GET /api/models`
- On selection change, calls `GET /api/architecture/{model_id}`
- Passes normalized architecture data down to the canvas component

#### `ArchitectureCanvas.tsx`
- Wrapper around `@excalidraw/excalidraw`
- Must use dynamic import with `ssr: false` (Excalidraw requires browser APIs)
- Receives architecture data as a prop
- Calls `diagram-generator.ts` to convert data into Excalidraw elements
- Uses `excalidrawAPI.updateScene()` to render/update the diagram

```tsx
import dynamic from "next/dynamic";

const Excalidraw = dynamic(
  () => import("@excalidraw/excalidraw").then((mod) => mod.Excalidraw),
  { ssr: false }
);
```

### Key Excalidraw API Usage

1. **`ExcalidrawElementSkeleton`** — simplified element creation. Only need to specify `type`, `x`, `y`, `width`, `height`, and optionally `label` (for text), `backgroundColor`, `strokeColor`. No need to generate IDs, versions, or other internal fields.

2. **`convertToExcalidrawElements(skeletons)`** — takes the simplified skeletons and produces full `ExcalidrawElement[]` with all required internal fields populated.

3. **`excalidrawAPI.updateScene({ elements })`** — replaces the current scene with new elements. Called whenever the user selects a different model.

---

## 4. Diagram Generation Strategy (`diagram-generator.ts`)

### Layout

Vertical stack, top to bottom. Each architectural component is a colored rectangle with a label and parameter annotation. Arrows connect consecutive blocks.

### Target Diagram (example: LLaMA 3 8B)

```
┌──────────────────────────────────┐
│  Input Text                      │
├──────────────────────────────────┤
│  Token Embedding                 │  128256 x 4096
├──────────────────────────────────┤
│  Positional Encoding (RoPE)      │  theta=500000
├──────────────────────────────────┤
│  Transformer Block  (x32)        │
│  ┌────────────────────────────┐  │
│  │ Multi-Head Self-Attention  │  │  32 heads, GQA: 8 KV
│  ├────────────────────────────┤  │
│  │ Add & RMSNorm             │  │
│  ├────────────────────────────┤  │
│  │ FFN (SiLU)                │  │  4096 → 14336 → 4096
│  ├────────────────────────────┤  │
│  │ Add & RMSNorm             │  │
│  └────────────────────────────┘  │
├──────────────────────────────────┤
│  Final RMSNorm                   │
├──────────────────────────────────┤
│  LM Head (Output Linear)        │  4096 → 128256
└──────────────────────────────────┘
```

### Element Generation Logic

The function `generateDiagramElements(config: ModelArchitecture)` returns `ExcalidrawElementSkeleton[]`:

1. **Input block** — light gray rectangle at top
2. **Embedding block** — blue rectangle, annotation: `{vocab_size} x {hidden_size}`
3. **Positional encoding block** — teal rectangle
   - If `rope`: show `"RoPE (theta={rope_theta})"`
   - If `learned`: show `"Learned Positional Embeddings (max={max_position_embeddings})"`
4. **Transformer block container** — large orange-bordered rectangle with `"x{num_hidden_layers}"` label
5. **Attention sub-block** — red rectangle inside transformer block
   - Annotation varies: `"{num_attention_heads} heads, MHA"` or `"GQA: {num_kv_heads} KV heads"` etc.
6. **Add & Norm sub-block** — light green, shows `"RMSNorm"` or `"LayerNorm"`
7. **FFN sub-block** — purple rectangle
   - Annotation: `"{hidden_size} → {intermediate_size} → {hidden_size}"`
   - Label includes activation: `"FFN ({activation})"`
8. **Second Add & Norm** — same as step 6
9. **Final normalization** — green rectangle
10. **LM Head** — dark blue rectangle, annotation: `"{hidden_size} → {vocab_size}"`

### Spacing & Positioning Constants

```typescript
const BLOCK_WIDTH = 400;
const BLOCK_HEIGHT = 60;
const INNER_BLOCK_WIDTH = 360;
const INNER_BLOCK_HEIGHT = 50;
const GAP = 20;           // vertical gap between blocks
const INNER_GAP = 10;     // gap between sub-blocks inside transformer
const START_X = 100;
const START_Y = 50;
const ANNOTATION_OFFSET = BLOCK_WIDTH + 30;  // x offset for parameter text
```

### Color Scheme

| Block | Background Color |
|---|---|
| Input | `#f5f5f5` (light gray) |
| Embedding | `#dbeafe` (light blue) |
| Positional Encoding | `#ccfbf1` (light teal) |
| Transformer Container | `#fff7ed` (light orange) |
| Attention | `#fee2e2` (light red) |
| Add & Norm | `#dcfce7` (light green) |
| FFN | `#f3e8ff` (light purple) |
| Final Norm | `#dcfce7` (light green) |
| LM Head | `#dbeafe` (light blue) |

### Arrows

Each block connects to the next via an `arrow` element skeleton:
```typescript
{
  type: "arrow",
  x: START_X + BLOCK_WIDTH / 2,
  y: currentY,
  width: 0,
  height: GAP,
  start: { id: previousElementId },
  end: { id: nextElementId },
}
```

---

## 5. File Structure

```
web/
├── backend/
│   ├── app.py                # FastAPI app, CORS config, endpoint definitions
│   ├── config_loader.py      # HuggingFace config fetch + field normalization
│   ├── models.py             # Pydantic schemas (ModelArchitecture, ModelListItem)
│   └── requirements.txt      # fastapi, uvicorn, transformers, pydantic
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx    # Root layout
    │   │   └── page.tsx      # Main page: ModelSelector + ArchitectureCanvas
    │   ├── components/
    │   │   ├── ModelSelector.tsx
    │   │   └── ArchitectureCanvas.tsx
    │   ├── lib/
    │   │   ├── api.ts              # fetch wrappers for backend endpoints
    │   │   └── diagram-generator.ts # config → ExcalidrawElementSkeleton[]
    │   └── types/
    │       └── model.ts            # TypeScript interfaces matching backend schemas
    ├── package.json
    ├── tsconfig.json
    └── next.config.js
```

---

## 6. Dependencies

### Backend (`requirements.txt`)

```
fastapi>=0.110.0
uvicorn>=0.29.0
transformers>=4.40.0
pydantic>=2.0.0
```

Note: `transformers` is used **only** for `AutoConfig.from_pretrained()` which downloads the tiny `config.json` file (~1-2KB). It does **not** download model weights.

### Frontend (`package.json` key deps)

```json
{
  "dependencies": {
    "next": "^15.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "@excalidraw/excalidraw": "^0.18.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/react": "^19.0.0",
    "@types/node": "^22.0.0"
  }
}
```

---

## 7. Dev Commands (Makefile Targets)

```makefile
# Install all dependencies
web-setup:
	cd web/backend && pip install -r requirements.txt
	cd web/frontend && npm install

# Start FastAPI backend on port 8000
web-backend:
	cd web/backend && uvicorn app:app --reload --port 8000

# Start Next.js frontend on port 3000
web-frontend:
	cd web/frontend && npm run dev

# Start both (requires two terminals or use &)
web-dev: web-backend web-frontend
```

---

## 8. API Contract Examples

### `GET /api/models`

```json
[
  { "model_id": "openai-community/gpt2", "display_name": "GPT-2 (124M)" },
  { "model_id": "openai-community/gpt2-xl", "display_name": "GPT-2 XL (1.5B)" },
  { "model_id": "meta-llama/Meta-Llama-3-8B", "display_name": "LLaMA 3 8B" },
  { "model_id": "Qwen/Qwen3-8B", "display_name": "Qwen3 8B" },
  { "model_id": "mistralai/Mistral-7B-v0.1", "display_name": "Mistral 7B" },
  { "model_id": "google/gemma-2b", "display_name": "Gemma 2B" }
]
```

### `GET /api/architecture/openai-community/gpt2`

```json
{
  "model_id": "openai-community/gpt2",
  "display_name": "GPT-2 (124M)",
  "architecture": {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_kv_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 50257,
    "max_position_embeddings": 1024,
    "attention_type": "mha",
    "positional_encoding": "learned",
    "rope_theta": null,
    "normalization": "layernorm",
    "activation": "gelu",
    "total_params": 124000000
  }
}
```

---

## 9. Key Design Decisions

1. **Config-only fetching**: We use `AutoConfig.from_pretrained()` which downloads only `config.json` (~1-2KB), never model weights. This keeps the backend fast and lightweight.

2. **Backend normalization**: Field name mapping happens server-side so the frontend always works with a single consistent schema regardless of model family.

3. **Excalidraw skeletons over raw elements**: Using `ExcalidrawElementSkeleton` + `convertToExcalidrawElements()` avoids manually generating UUIDs, version numbers, and other internal Excalidraw bookkeeping.

4. **Dynamic import for Excalidraw**: Excalidraw depends on browser APIs (`window`, `document`) and must be loaded client-side only via Next.js dynamic import with `ssr: false`.

5. **In-memory caching**: Model configs are immutable for a given model ID. Cache them in the backend to avoid repeated HuggingFace API calls.

---

## 10. Future Extensions

- **Side-by-side comparison**: Select two models and render diagrams side by side with differences highlighted
- **Per-layer parameter counts**: Show parameter count for each sub-block (attention weights, FFN weights, etc.)
- **Export**: PNG/SVG export using Excalidraw's built-in export utilities
- **Data flow animation**: Animate a token flowing through the architecture
- **Link to implementation**: Click a diagram block to jump to the corresponding code in this repo's tokenizer/embedder/transformer modules
- **Custom model support**: Paste a HuggingFace model ID not in the curated list
