"""Show the prompts training will actually see.

Printed once whenever training starts or resumes, and runnable standalone:

    python -m underfit.training.prompt_preview \
        --dataset-config runs/<run>_dataset.json \
        --model-config   runs/<run>_model.json

Samples are pulled through the *same* dataset object the DataLoader feeds the
loop, so templates, trigger tokens, shuffling and fallbacks are whatever
training really gets. Nothing here rebuilds a prompt — that is the whole point:
a preview that re-implemented the pipeline could agree with itself while
disagreeing with training.

It also reports when a dataset config carries `prompt_config` that never
reached the metadata module, which is a silent failure otherwise: the module
falls back to legacy tag prompts (or to an empty string when the clips have no
tags) and training looks fine while ignoring everything the dashboard
configured.
"""

import argparse
import json
import random
import sys

PREVIEW_COUNT = 10


def _unwrap(fn):
    """Datasets that feed DataLoader workers store the metadata fn dill-pickled.
    Unpickling is how we inspect what a worker will actually run — which is the
    only copy that matters, since it carries its own snapshot of the module
    globals taken when the dataloader was built."""
    if isinstance(fn, (bytes, bytearray)):
        try:
            import dill
            return dill.loads(fn)
        except Exception:
            return None
    return fn


def _metadata_fns(dataloader):
    """The custom_metadata_fn objects actually installed on the dataset.

    Reached through the dataset rather than reconstructed, so this reflects the
    live wiring. Layouts differ per backend, hence the duck-typing.
    """
    ds = getattr(dataloader, "dataset", None)
    if ds is None:
        return []
    fns = []
    # PreEncodedDataset keeps a list; other datasets keep one; some backends
    # hang them off per-dataset config objects.
    for attr in ("custom_metadata_fns", "custom_metadata_fn"):
        v = getattr(ds, attr, None)
        if v is None:
            continue
        if isinstance(v, dict):
            fns.extend(_unwrap(x) for x in v.values() if x is not None)
        elif isinstance(v, (list, tuple)):
            fns.extend(_unwrap(x) for x in v if x is not None)
        elif callable(v) or isinstance(v, (bytes, bytearray)):
            fns.append(_unwrap(v))
    for attr in ("configs", "datasets", "dataset_configs"):
        for cfg in (getattr(ds, attr, None) or []):
            fn = getattr(cfg, "custom_metadata_fn", None)
            if fn is not None:
                fns.append(fn)
    return fns


def _configure_fn(fn, dataset_config):
    """Set prompt_config on a metadata fn's module globals. True if it took."""
    g = getattr(fn, "__globals__", None)
    if not g or "_prompt_config" not in g:
        return False
    setter = g.get("set_config")
    if callable(setter):
        # Call it for any side effects beyond the assignment.
        try:
            setter(dataset_config)
        except Exception:
            pass
    # Then assign directly, and do not trust the setter to have done it. dill
    # rebuilds each function with its own globals mapping, so after an unpickle
    # set_config.__globals__ is NOT the same dict as get_custom_metadata.__globals__:
    # the setter writes _prompt_config into a namespace the dataset never reads.
    # Assigning here targets the mapping the metadata fn itself resolves against.
    g["_prompt_config"] = dataset_config.get("prompt_config")
    return g.get("_prompt_config") is not None


def ensure_prompt_config(dataloader, dataset_config):
    """Hand prompt_config to the metadata module the dataset will actually run.

    The sa3 backend configures the module as it loads it. stable-audio-tools does
    not: its factory runs its own spec_from_file_location + exec_module per dataset,
    so the module it uses is a different object from anything underfit loaded and a
    set_config() call on our copy never reaches it. That left every sat-backend run
    on legacy tag prompts -- empty strings on clips with no tags, i.e. unconditional
    training -- with everything configured in NEW FINETUNE silently discarded.

    The fn is not stored as a function. SAT keeps it dill-pickled
    (``custom_metadata_fns[path] = dill.dumps(fn)``) and dill.loads a fresh copy on
    every __getitem__, so the pickle carries its own frozen snapshot of the module
    globals. Mutating an unpickled copy changes nothing; the bytes have to be
    re-dumped, which is what this does.

    Returns the number of entries reconfigured.
    """
    if not isinstance(dataset_config, dict) or not dataset_config.get("prompt_config"):
        return 0
    ds = getattr(dataloader, "dataset", None)
    if ds is None:
        return 0
    n = 0
    store = getattr(ds, "custom_metadata_fns", None)
    if isinstance(store, dict):
        for key, raw in list(store.items()):
            if isinstance(raw, (bytes, bytearray)):
                try:
                    import dill
                    fn = dill.loads(raw)
                    if _configure_fn(fn, dataset_config):
                        store[key] = dill.dumps(fn)      # re-freeze, or it is lost
                        n += 1
                except Exception:
                    continue
            elif callable(raw) and _configure_fn(raw, dataset_config):
                n += 1
    for attr in ("custom_metadata_fn",):
        fn = getattr(ds, attr, None)
        if callable(fn) and _configure_fn(fn, dataset_config):
            n += 1
    for attr in ("configs", "datasets", "dataset_configs"):
        for cfg in (getattr(ds, attr, None) or []):
            fn = getattr(cfg, "custom_metadata_fn", None)
            if callable(fn) and _configure_fn(fn, dataset_config):
                n += 1
    return n


def diagnose_prompt_config(dataloader, dataset_config):
    """Return a warning string when prompt_config never reached the module."""
    if not isinstance(dataset_config, dict):
        return None
    pc = dataset_config.get("prompt_config")
    if not pc:
        return None
    for fn in [f for f in _metadata_fns(dataloader) if f is not None]:
        # prompt_templates.py keeps its config in a module global that
        # set_config() populates; if it's still None the config was dropped.
        g = getattr(fn, "__globals__", {})
        if "_prompt_config" in g and g["_prompt_config"] is None:
            return (
                "dataset config carries prompt_config "
                f"({', '.join(sorted(pc))}) but the metadata module never "
                "received it — set_config() was not called, so prompts fall "
                "back to legacy tag strings (empty when clips have no tags). "
                "Everything configured in NEW FINETUNE is being ignored."
            )
    return None


def collect_prompts(dataloader, n=PREVIEW_COUNT, seed=None):
    """Pull n samples through the real dataset and return their metadata.

    Returns a list of dicts. Repeats across the same clip are expected and
    useful on small datasets: they show the per-sample randomisation training
    sees, rather than one frozen example.
    """
    ds = getattr(dataloader, "dataset", None)
    out = []
    rng = random.Random(seed)

    if ds is not None and hasattr(ds, "__getitem__") and hasattr(ds, "__len__"):
        try:
            size = len(ds)
        except Exception:
            size = 0
        if size:
            for _ in range(n):
                i = rng.randrange(size)
                try:
                    item = ds[i]
                except Exception as e:
                    out.append({"error": f"{type(e).__name__}: {e}"})
                    continue
                md = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else {}
                out.append(md if isinstance(md, dict) else {})
            return out

    # Iterable datasets (e.g. WebDataset) have no indexing — take one batch.
    try:
        _reals, metadata = next(iter(dataloader))
        for md in list(metadata)[:n]:
            out.append(md if isinstance(md, dict) else {})
    except Exception as e:
        out.append({"error": f"could not read a batch: {type(e).__name__}: {e}"})
    return out


def _decoder(tokenizers):
    """Return a fn that turns a tokenized prompt back into text.

    The training loop passes `tokenizers` into create_dataloader, so the dataset
    replaces metadata["prompt"] with the tokenizer output — the preview would
    otherwise print input_ids tensors. Decoding is also more faithful than
    reading the pre-tokenisation string: it shows the text after truncation to
    the conditioner's max_length, which is what the model is actually
    conditioned on.
    """
    toks = []
    for v in (tokenizers or {}).values():
        tok = v[0] if isinstance(v, (tuple, list)) and v else v
        if hasattr(tok, "decode"):
            toks.append(tok)
    if not toks:
        return None
    tok = toks[0]

    def decode(value):
        ids = value
        if isinstance(value, dict):
            ids = value.get("input_ids")
        if ids is None:
            return None
        try:
            if hasattr(ids, "dim") and ids.dim() > 1:
                ids = ids[0]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            return None

    return decode


def _prompt_text(md, decode):
    """The prompt as text, whether the dataset tokenized it or not."""
    value = md.get("prompt")
    if isinstance(value, str) or value is None:
        return value
    if decode is not None:
        text = decode(value)
        if text is not None:
            return text
    return f"<tokenized, no tokenizer to decode: {type(value).__name__}>"


def _fmt(value, limit=160):
    if value is None:
        return "<missing>"
    s = str(value)
    if s == "":
        return "<EMPTY STRING>"
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def print_prompt_preview(dataloader, dataset_config=None, n=PREVIEW_COUNT, seed=None,
                         tokenizers=None):
    """Print n example prompts exactly as training will receive them."""
    print(f"[prompts] {n} example prompts as the training loop will see them:", flush=True)
    rows = collect_prompts(dataloader, n=n, seed=seed)
    decode = _decoder(tokenizers)
    empty = 0
    for i, md in enumerate(rows, 1):
        if md.get("error"):
            print(f"  {i:2}. <error> {md['error']}", flush=True)
            continue
        prompt = _prompt_text(md, decode)
        if prompt == "" or prompt is None:
            empty += 1
        src = md.get("src_relpath") or md.get("relpath") or ""
        print(f"  {i:2}. {_fmt(prompt)}", flush=True)
        if src:
            print(f"      └ {_fmt(src, 100)}", flush=True)
    if empty:
        print(f"[prompts] WARNING: {empty}/{len(rows)} sampled prompts are empty — "
              f"training is (partly) unconditional.", flush=True)
    warn = diagnose_prompt_config(dataloader, dataset_config)
    if warn:
        print(f"[prompts] WARNING: {warn}", flush=True)
    return [{**md, "prompt_text": _prompt_text(md, decode)} for md in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Preview the prompts training will see")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("-n", "--count", type=int, default=PREVIEW_COUNT)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    from underfit.backends import get_backend

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    with open(args.model_config) as f:
        model_config = json.load(f)

    backend = get_backend(args.backend)
    # Same call the training loop makes, minus the perf knobs — batch_size 1 and
    # no workers keep this cheap and keep tracebacks in this process.
    dataloader = backend.create_dataloader(
        dataset_config,
        batch_size=1,
        num_workers=0,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )
    rows = print_prompt_preview(dataloader, dataset_config, n=args.count, seed=args.seed)
    return 1 if all((r.get("prompt_text") in ("", None)) for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
