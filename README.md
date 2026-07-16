# underfit

**The power-user LoRA dashboard for making custom finetunes of [Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-medium).**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dada-bots/underfit/blob/main/underfit-colab.ipynb)  ![python](https://img.shields.io/badge/python-3.10-blue)  ![license](https://img.shields.io/badge/license-MIT-green)

---

<img width="704" height="908" alt="underfit-dashboard-screenshot" src="https://github.com/user-attachments/assets/2addccd6-de7b-4af8-a7b2-b9e61b2de70f" />



## Quickstart — local GPU box

Linux box with an NVIDIA GPU, `git`, and `curl`. Everything else gets fetched automatically.

```bash
git clone https://github.com/dada-bots/underfit && cd underfit
./install.sh                            # ~5 min: installs uv, syncs deps, clones SA3, downloads model packs
./run.sh                                # serves the dashboard on http://localhost:8787
```

Open the URL in a browser, click **+ New Dataset** to star

> **First time only** — open https://huggingface.co/stabilityai/stable-audio-3-medium and click *Agree and access repository*. The three SA3 ARC repos share one license; one click unlocks all three. Approval is instant. Without it the install fails with `401 Unauthorized` on the ARC checkpoint downloads. (The base checkpoints aren't gated.)

> To reach the dashboard in your browser If your linux box is remote, you'll need to either tunnel to localhost:8787 over ssh, or use NGROK (recommended) to forward a domain to http://localhost:8787 


---


## Quickstart — Apple Silicon (Mac, MLX)

On an Apple-Silicon Mac, underfit trains on the **Metal GPU via MLX** — no CUDA,
no PyTorch training loop. It drives the pure-MLX SA3 trainer in the sibling
[stable-audio-3](https://github.com/Stability-AI/stable-audio-3) `optimized/mlx`
runtime, which is fast (an M4 Pro finetunes the medium model at a usable clip).

```bash
# 1. Clone underfit and stable-audio-3 as siblings
git clone https://github.com/dada-bots/underfit
git clone https://github.com/Stability-AI/stable-audio-3

# 2. Set up the MLX runtime — its own venv + the MLX weight packs (base + ARC + codec + T5)
cd stable-audio-3/optimized/mlx && ./install.sh && cd -

# 3. Install the dashboard's Python deps (skip the torch-backend wizard — unused on MLX)
cd underfit && ./install.sh --no-setup

# 4. Launch with the MLX engine (auto-finds the sibling optimized/mlx checkout)
UNDERFIT_ENGINE=mlx ./run.sh          # dashboard on http://localhost:8787
```

Then drive the dashboard exactly as on Linux, with a few Apple niceties:

- The **Engine** dropdown in *New Finetune* shows **MLX** (MPS/torch is hidden
  unless a torch backend is installed).
- Dataset **encoding is torch-free** — the MLX SAME encoder, no torch checkpoint.
- If a model's MLX weights aren't present, *New Finetune* / *Encode* offer a
  **one-click download** into the MLX checkout (base + ARC pair together).
- For **SA3-medium** the demo defaults are tuned for MLX — **ARC-only, 8-step,
  SAME-S decode** — so demos render in seconds instead of the ~8 min a 50-step
  full-length RF demo would take.

If `stable-audio-3/` isn't a sibling of `underfit/`, point at it explicitly:
`UNDERFIT_MLX_ROOT=/path/to/stable-audio-3/optimized/mlx` (and
`UNDERFIT_MLX_PYTHON` for its venv's python).

> Prefer the raw trainer CLI without the dashboard? The MLX runtime ships a
> standalone LoRA-training CLI — see
> [optimized/mlx → LoRA training](https://github.com/Stability-AI/stable-audio-3/tree/main/optimized/mlx#lora-training).

---

## Quickstart — Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dada-bots/underfit/blob/main/underfit-colab.ipynb) 

Jankier than running locally, but still works. Read more here: https://github.com/dada-bots/underfit#running-on-colab

---

## Requirements

|                | |
|---             |---|
| **OS**         | Linux (NVIDIA) or macOS (Apple Silicon, via **MLX**). Windows untested. |
| **GPU**        | NVIDIA — ≥16 GB VRAM ideal, 8 GB works minimally (fp16 base + low rank + small latent crop + batch 1). Or **Apple Silicon** (M1–M4) on the MLX engine — unified memory, 16 GB+ recommended for medium. |
| **Python**     | 3.10 (auto-fetched by `uv`). |
| **Disk**       | Plan for ~17 GB per SA3-medium pack, ~7 GB per small pack. All three = ~31 GB of checkpoints. Datasets add a few hundred MB each. |
| **HF account** | Free. Needed to accept the [SA3 license](https://huggingface.co/stabilityai/stable-audio-3-medium) once. |

| Hardware   | Fit | Notes |
|---         |---|---|
| H100 (80GB)| ✅ | Great. Very fast. Holds many finetunes per GPU |
| A100 / L4  / G4 / 4090 | ✅ | Comfortable. |
| T4 (16GB)  | ⚠️ | Slow, requires patience, but works. |
| **Apple Silicon (M1–M4)** | ✅ | Trains on the Metal GPU via the **MLX engine** — see the [Apple-Silicon quickstart](#quickstart--apple-silicon-mac-mlx). Fast on M-series (an M4 Pro finetunes medium at a usable clip). |
| Mac (Intel) / MPS-torch | ❌ | No MLX (needs Apple Silicon). The torch/MPS training loop exists but is far too slow to recommend. |
| CPU only   | ❌ | Won't train. |

---





## Underfit Features

- **Encode any folder of audio** into pre-encoded latents — multi-GPU sharded if you have more than one card. Supported: WAV / FLAC / MP3 / OGG / OPUS / M4A / AIFF.
- **Manage dozens of LoRA finetunes simultaneously** with live loss curves and a "loss-by-timestep" chart that splits the loss across the diffusion sigma range. Demo MP3s + tinted-spectrogram previews get generated every N steps.
- **Launch Gradio inference UIs per checkpoint** with a [LAUNCH] button. Dial LoRA strength <1.0 to blend with the base model, skip the LoRA on early denoising steps for cleaner song structure, do audio2audio style transfer, or inpaint regions of an input.
- **Blend multiple LoRAs at once** via `run_gradio.py --lora-ckpt-path lora_a.safetensors lora_b.safetensors`.
- **Continue training from a previous LoRA** — upload a `.safetensors` in *New Finetune* and the run picks up where it left off. Lets you mix datasets / styles iteratively.
- **Auto-resume across disconnects** — runs are detached subprocesses managed by the dashboard's `RunsRegistry`; the dashboard re-reads `runs.json` on launch so closing the tab or restarting the server doesn't kill your training.

---

## Adapter formats

> Throughout this README **LoRA** is used as a catch-all term for the whole adapter family — DoRA, BoRA, the -XS variants, and original LoRA included. 

| Format        | What it adds | When |
|---            |---|---|
| **LoRA**      | `lora_A`, `lora_B` | The original low-rank adapter. Inferior, but well-understood. |
| **DoRA**      | LoRA + a per-column magnitude vector | **Recommended default.** Generally better-quality fits than vanilla LoRA. |
| **BoRA**      | LoRA + per-row *and* per-column magnitudes | Like DoRA but scales in both dimensions. |
| **LoRA-XS / DoRA-XS / BoRA-XS** | Same as above, but the rank-`r` matrices are factored against fixed SVD bases instead of being free parameters | "Extra-small" variants. Smaller files, slightly less capacity. |

Pick one in the **LoRA type** dropdown of *New Finetune*. All variants produce a single `.safetensors` file you can load anywhere.

---

## Models supported

| Pack | HF repo | Trained Duration | Best for |
|---|---|---|---|
| `sa3-medium`   | [`stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium)             | 6:20 | General music + SFX. Best quality. ~17 GB. |
| `sa3-sm-music` | [`stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)   | 2:00 | Music-focused, small. ~7 GB. |
| `sa3-sm-sfx`   | [`stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)       | 2:00 | SFX-focused, small. ~7 GB. |

Each pack ships in two parts:
- **Base** (rectified-flow) — 50 step model.
- **ARC** (adversarial-consistency) — 8 step accelerated model. This the base model with adversarial post-training. 

Note: in order to download the models, you need to agree to the HuggingFace license on the repo pages.

Important: we only do LoRA training on the Base model. However, by miracle, LoRAs trained on Base work perfectly with ARC during inference. During training, underfit generates demos with both. After training, you probably want to use the ARC model for inference. 

---

## Step-by-step: your first LoRA

### 1. Get your audio in a folder

10+ minutes of audio is the floor; 30+ min is better. **Quality > quantity.** One coherent style per dataset (one artist, one genre, one SFX category). Mixed bags train into mush. The dashboard lets you tick/untick individual files after the scan, so you don't have to pre-curate. 

### 2. (Optional) Add metadata for prompts

Each clip can have key-value metadata that the LoRA learns to associate with the audio. underfit looks in this order:

**a. JSON sidecar** A file with the same name as the audio, `.json` extension, sitting either:

- right next to the audio: `my-songs/01.wav` + `my-songs/01.json`, or
- in a sibling `json/` folder: `my-songs/01.wav` + `my-songs/json/01.json`

Any string/number values are used

```json
{
  "title": "intrauterine fetal demise",
  "genre": "technical death metal",
  "bpm": 145,
  "mood": "absurd",
}
```

**b. Plain `.txt` sidecar** — the [Stable Audio 3 convention](https://github.com/Stability-AI/stable-audio-3/blob/main/scripts/train_lora.py). Same stem as the audio, `.txt` extension, sitting either next to the audio (`my-songs/01.wav` + `my-songs/01.txt`) or in a sibling `txt/` folder (`my-songs/01.wav` + `my-songs/txt/01.txt`). The whole file content (whitespace-stripped) becomes the `prompt` metadata key. Simplest possible per-clip captioning — and if you already have a dataset prepared for SA3's own `pre_encode_dataset.py` script, it Just Works as-is.

**c. Embedded audio tags** (ID3 on MP3, Vorbis on FLAC/OGG, M4A atoms, etc.) — read via the `audio_metadata` library. For example: `title`, `artist`, `album`, `genre`, `label`, `date`, `composer`, `bpm`. If your music library is already tagged in Picard / Mp3tag, you get those for free.

**d. Skip metadata entirely** — totally fine. In *Configure prompts* you can compose prompts from the file path (e.g. `/sfx/explosions/fireworks/03.wav` becomes the prompt — folder structure becomes meaning), or a fixed string for every clip, or both mixed.

### 3. Create a dataset

In the dashboard click **+ Dataset**. Paste the path to your audio folder. The dashboard scans, lets you tick/untick files, then spawns a pre-encoding subprocess on the GPU. Each audio file becomes a `.npy` (encoded latent) + `.json` (metadata) pair. When the dataset appears in the **Datasets** panel it's ready to train against.

> On a T4 this can take a while. The encoding panel updates as each file finishes.

### 4. New Finetune — fill in the form

In the dashboard click **+ Finetune**.

| Field | What to put | Why |
|---|---|---|
| **Name** | `my-first-lora` (alphanumeric + hyphens) | Run ID + `.safetensors` filename |
| **Model** | `sa3-medium` | Base model to finetune against |
| **Dataset** | the one you created in Step 3 | Pre-encoded latents |
| **LoRA type** | **`DoRA`** | Recommended. |
| **LoRA rank** | `16` | Capacity. Higher = more parameters + sometimes higher quality + more overfitting risk. Smaller = sometimes learns style better. |
| **Steps** | `20000` | A reasonable LoRA lands around 10k — that's where it *creatively underfits*: still varied on new prompts, not yet memorising. Past 20k it may overfit. |
| **Batch size** | `1` on T4, up to `8` on H100 | Bigger = uses more VRAM. |
| **Latent length** | model default, or shorter | You can train shorter than the model's max. Lower = faster, less memorization, often better style learning, useful for audio2audio. |
| **Learning rate** | leave default | Higher may learn faster but also collapse training. May need higher for XS variants. Lower may help learn more subtle details. |
| **Demo every** | `500`–`1000` | Frequency of demo audio generation during training. |
| **Checkpoint every** | `500` | Frequency of `.safetensors` saves. Each one is restartable. |

**Tips:**

- **Latent length is the underrated knob.** Lowering it to ~47 s or ~12 s (with `random_crop` on) is often the cleanest way to learn a style *without* memorisation. The model only ever sees patterns at that timescale and never sees full songs, so it can't memorise structure. Listen to the demos at full length to hear how it extrapolates.
- **Batch size is a creative parameter.** Just because you have VRAM for 8 doesn't mean you want 8. `batch_size=1` learns something different (focuses on one song at a time, sharper imprint) from `batch_size=4` (averages gradients across songs, smoother fit). Experiment.
-  **Demo every for streamable performance** A properly tuned value (like 250 on an H100) will train faster than you can listen to the demos. With automatic playthrough you can hear it continously as it trains. 

### 5. Pick a GPU

Each GPU card shows current VRAM use + an estimate of how much your run will need (based on model + rank + batch). If the estimate goes red, lower batch size or rank. The wizard pins the chosen GPU via `CUDA_VISIBLE_DEVICES` automatically.

### 6. Configure prompts

How prompts get built from your dataset each training step. Three sources with balance percentages that should sum to 100:

- **Tags** — the metadata fields from Step 2 (album / year / genre / bpm / …). Toggle which keys to include.
- **Paths** — directory + filename. Useful when your folder structure encodes meaning (e.g. `metal/death/archspire/`).
- **Fixed** — a literal string applied to every file. Use for single-style datasets where every example should share the same prompt.

Other knobs:
- **Trigger word** — optionally prepend a magic phrase to N% of prompts.
- **Shuffle** — randomize the order of comma-separated parts each step (keeps the model from memorising comma order).

**Simplest config for a small single-style dataset:** Fixed text = your trigger phrase, balance 100 % fixed.

**Good mix to try:** ~50/50 fixed + tags (or fixed + paths). Training sees both the shared style anchor *and* per-song detail (title, BPM, genre, etc.). At inference you can either use just the trigger (style only) or trigger + per-song details (style on a specific kind of song).

**You're blending two prompt vocabularies.** SA3's base model was trained on labelled key-value strings like `Genre: techno, BPM: 140, Mood: dark`. Your LoRA prompts get composed *on top* of that vocabulary at inference time. Re-using the base model's format seems to help the LoRA stack with what the base already knows, but you can also combine old and new (e.g. `/metal/song.mp3, BPM: 140`). Skim a few of your demo prompts to spot-check the format.

### 7. Set up demos

The MP3s + spectrograms generated for you to listen to during training. You can:
- Edit individual demos: prompt text, sampler (RF or ARC), CFG, step count, seed.
- Re-roll prompts with the per-demo `↻` or the big `↻` to re-sample from the dataset.
- Add / remove demos with `+` / `−`.

### 8. Launch and what to watch

Click **Launch**. Then:
1. **Run appears** in the run panel with status `loading…` (while the base model loads into VRAM).
2. Status flips to `training`; **loss curve** starts plotting.
3. Every `demo_every` steps the run pauses to generate **demo MP3s** with tinted-stereo spectrogram previews.
4. Every `checkpoint_every` steps a fresh `.safetensors` lands in the **Checkpoints** list.

### 9. How to know when to stop

**Loss curve.** Watch the loss panel. The *elbow* — where the loss stops being initially flat and begins to drop — tends to be the most *creatively underfit* checkpoint and my favorite to keep. Past the elbow you're creeping toward memorization.

**Your ears, on the demos.** Open a demo MP3 every couple thousand steps. What you're listening for:

- **Base RF demos (CFG≈7) light up first.** Around the time the run is "getting it," your CFG=7 demos suddenly sound right — clearly your style on a coherent prompt.
- **Then CFG=7 over-cooks, and CFG=1 takes over.** Past the elbow, CFG=7 starts sounding artifacted / over-saturated. The lower-CFG demos (CFG≈1) keep improving and end up sounding cleanest. If CFG=1 sounds good and CFG=7 doesn't, that's a sign the LoRA has internalised the style and no longer needs the prompt-classifier guidance.
- **Conditional → unconditional crossover.** Early on, only prompted demos sound like the training style. Later, even **empty-prompt** demos start sounding like the style — the model has *absorbed* the dataset.
- **ARC demos lag a bit but end up cleaner.** ARC-distilled demos take a few thousand more steps to catch up to base RF, but final quality is usually better.

**Don't fear a memorized checkpoint.** "Overfitting" is only a problem if you're chasing creative variation. A memorized checkpoint can still be useful:
- You can weaken the **LoRA strength** at inference
- **audio2audio / style-transfer** effects often hit harder with a memorized model — the strong style signal pulls input audio into the training distribution more decisively.

Save checkpoints liberally. Different downstream uses want different points on the underfit ↔ memorise curve.

You can stop a run anytime — the last checkpoint is yours. You can also resume from any checkpoint to train further.

### 10. Get the `.safetensors` out

In the dashboard's checkpoints list, click the download (⬇) button — or grab it directly from disk at `state/runs/<run-id>/<step>.safetensors`. Drop it into any SA3 inference setup and the adapter layer-grafts onto the base model at runtime.

---

## Run inference with your LoRA

### From the dashboard

Click [Launch] on any checkpoint. Spins up a private Gradio link. Knobs:

- **LoRA strength <1.0** — blends LoRA with base model. A heavily-trained / "memorised" checkpoint often sounds best around 0.6–0.8 — gives you "in the style of" without straight regurgitation.
- **LoRA interval (skip first step)** — the very first denoising step establishes song structure, so skipping the LoRA here prevents it from regurgitating any song. Instead the base model determines the song structure fron the prompt. 
- **audio2audio** — drop in a song; SA3 + your LoRA stylize it.
- **Inpainting** — paint over a region of an input and regenerate just that region in the LoRA's style.

### From the CLI

```bash
.venv/bin/python run_gradio.py \
    --model-config state/models/sa3-medium/base/model_config.json \
    --ckpt-path    state/models/sa3-medium/base/model.safetensors \
    --lora-ckpt-path my-lora.safetensors                 # one or more
```

Pass multiple `.safetensors` to blend LoRAs. Strengths interact non-linearly — worth experimenting.

### Beyond the dashboard

- **Continue training from an uploaded LoRA.** In *New Finetune*, the *Start from a previous LoRA* upload picks up your `.safetensors` and keeps going. There's an art to mixing datasets this way — e.g. I trained a music-style DoRA (Dadabots, baroquecore) for 10k steps at 47s, then continued on a production-quality DoRA (Encanti, bass music) at 12s for 300 steps. The result was a LoRA that fuses both styles in a way neither alone produce.
- **Any SA3 inference setup can theoretically load it if they support it.** Plugins, ComfyUI nodes, custom Python wrappers.

---

## Running on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dada-bots/underfit/blob/main/underfit-colab.ipynb)

Same underfit, same dashboard, just hosted on Colab's GPU instead of yours. The notebook walks through everything end-to-end. Definitely jankier than running it locally yourself. 

### Why Colab

Zero install on your machine. Free GPU access for experimenting. Free tier is slow (T4). Pro tier ($9.99) gets you a H100 which is fast.

### Important Colab-specific quirks

**Pre-emption.** At some point your session will time-out and get canceled. Ephemeral storage will be deleted. However, Google Drive integration means your runs persist across sessions. On the next reboot, the backup logs, checkpoints, training runs will be available in the dashboard.

**Use ngrok in Step 4.** Colab's built-in port-proxy buffers HTTP responses aggressively, which makes the dashboard feel laggy and occasionally freeze. Audio playback is the worst offender — the proxy holds the entire audio file before forwarding to your browser instead of streaming it, blocking every other request until done. *Training itself is unaffected* — it runs as a detached subprocess on the GPU and survives dashboard freezes or closed tabs. Free signup at [ngrok.com](https://ngrok.com) and paste your auth token into the `NGROK_AUTHTOKEN` field in Step 4. If anything ever freezes, re-run Step 4 to restart the server. Colab is nice, but using a normal Linux box is the most reliable setup. 


### Colab troubleshooting

- **"NO GPU DETECTED" in Step 1** → Runtime → Change runtime type → Hardware accelerator → pick a GPU → Save. Re-run Step 1.
- **`401 Unauthorized` / `GatedRepoError` in Step 3** → Accept the SA3 license at https://huggingface.co/stabilityai/stable-audio-3-medium. One click unlocks all three ARC repos. Re-run Step 3.
- **Dashboard frozen** → re-run Step 4. Training runs are unaffected.

---

## Troubleshooting (general)

- **`GatedRepoError: 403 Client Error`** on ARC downloads → You haven't accepted the [SA3 license](https://huggingface.co/stabilityai/stable-audio-3-medium). One click, instant approval, all three ARC repos unlock.
- **`ModuleNotFoundError: No module named 'stable_audio_3'`** → The SA3 sibling clone got moved or never installed. Re-run `./install.sh` — it'll re-clone and `uv pip install -e` it.
- **Dashboard says port already in use** → another instance is running. Find with `ps aux | grep server.py`, kill it, then `./run.sh` again.
- **Demos sound identical to the input** → overfitting too fast. Go back and and find an earlier checkpoint that is more interesting. Re-run with a lower sequence length and random crop. Or lower the rank, or learning rate.
- **Demos sound nothing like the input even at 20k+** → the dataset may be too varied, the rank too low, or the LR too low.

---

## Project layout

```
underfit/             CLI + training loop + backend abstraction
dashboard/            webapp — server.py + index.html + per-model registries
dataset_processing/   pre-encode + autotagger
lora_train.py         trainer entrypoint
run_gradio.py         inference entrypoint (used by the [LAUNCH] button)
install.sh, run.sh    install + launch helpers
underfit-colab.ipynb  the Colab tutorial — same dashboard, hosted on Colab
```

`stable-audio-3/` (the SA3 backend) gets cloned next to `underfit/` during install. Multiple `underfit/` clones share one SA3 clone and one HF cache.

---

## Contributing

Issues and PRs welcome at [github.com/dada-bots/underfit/issues](https://github.com/dada-bots/underfit/issues).

When reporting a bug, paste the output of:

```python
from underfit.monitor import debug_info; debug_info()
```

That gives the maintainer everything useful — git HEAD, torch + CUDA build, GPU compute caps, the latest run's log + every sidecar, disk usage, an SA3 import check.

---

## Credits & license

[MIT](./LICENSE). underfit is a thin layer over Stable Audio (Please refer to the Stability Community License):

- **[Stable Audio 3 models](https://huggingface.co/stabilityai/stable-audio-3-medium)** by Harmonai — the diffusion model being finetuned
- ** [Stable Audio 3 repo](https://github.com/Stability-AI/stable-audio-3) the `sa3` backend, the minimal code for running the model.
- **[Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)** — the original training infrastructure that the `sat` backend mirrors

- LoRA family papers: [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al. 2021), [DoRA](https://arxiv.org/abs/2402.09353) (Liu et al. 2024)

Built by [@dadabots](https://dadabots.com).
