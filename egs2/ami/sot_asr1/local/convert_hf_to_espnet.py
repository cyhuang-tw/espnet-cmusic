#!/usr/bin/env python3
"""Convert HF Whisper (TS-ASR-Whisper) checkpoint to ESPnet SOT format.

Converts state dict keys from HuggingFace Transformers naming to native
OpenAI Whisper naming (as used by ESPnet's OpenAIWhisperEncoder/Decoder).

Handles:
- Systematic key renaming (HF → native Whisper conventions)
- Token embedding split (single embed_tokens → ori_emb + add_emb)
- DiCoW-specific weight filtering
- Optional model-level validation

Usage:
    python local/convert_hf_to_espnet.py \\
        --hf_checkpoint /path/to/model.safetensors \\
        --espnet_config conf/tuning/train_sot.yaml \\
        --token_list data/token_list.txt \\
        --output_pth exp/converted/model.pth \\
        [--added_tokens local/added_tokens.txt] \\
        [--validate]
"""

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import torch

WHISPER_BASE_VOCAB_DEFAULT = 51865  # multilingual Whisper base vocabulary size


def get_whisper_base_vocab(espnet_config_path: str) -> int:
    """Determine base vocab size from the ESPnet config's whisper_model.

    large-v3-turbo has 51866 base tokens; all other multilingual models
    have 51865.
    """
    import yaml

    with open(espnet_config_path) as f:
        config = yaml.safe_load(f)

    whisper_model = config.get("decoder_conf", {}).get("whisper_model", "")
    if "turbo" in whisper_model:
        return 51866
    return WHISPER_BASE_VOCAB_DEFAULT

# ──────────────────────────────────────────────────────────────────────
# Key renaming rules: (HF substring, ESPnet replacement)
# Applied sequentially — all matching rules fire in order.
# ──────────────────────────────────────────────────────────────────────
RENAME_RULES = [
    # ── Encoder top-level ──
    ("model.encoder.embed_positions.weight", "encoder.encoders.positional_embedding"),
    ("model.encoder.layer_norm.", "encoder.encoders.ln_post."),
    ("model.encoder.conv", "encoder.encoders.conv"),
    ("model.encoder.layers.", "encoder.encoders.blocks."),
    # ── Decoder top-level ──
    ("model.decoder.embed_positions.weight", "decoder.decoders.positional_embedding"),
    ("model.decoder.layer_norm.", "decoder.decoders.ln."),
    ("model.decoder.layers.", "decoder.decoders.blocks."),
    # ── Cross-attention (decoder only, must precede self-attention rules) ──
    (".encoder_attn.q_proj.", ".cross_attn.query."),
    (".encoder_attn.k_proj.", ".cross_attn.key."),
    (".encoder_attn.v_proj.", ".cross_attn.value."),
    (".encoder_attn.out_proj.", ".cross_attn.out."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    # ── Self-attention ──
    (".self_attn.q_proj.", ".attn.query."),
    (".self_attn.k_proj.", ".attn.key."),
    (".self_attn.v_proj.", ".attn.value."),
    (".self_attn.out_proj.", ".attn.out."),
    (".self_attn_layer_norm.", ".attn_ln."),
    # ── FFN ──
    (".fc1.", ".mlp.0."),
    (".fc2.", ".mlp.2."),
    (".final_layer_norm.", ".mlp_ln."),
]

# HF keys to skip entirely (DiCoW-specific or handled separately)
SKIP_PREFIXES = (
    "model.encoder.fddts.",
    "model.encoder.initial_fddt.",
    "model.encoder.spk_transforms.",
    "model.encoder.lm_head.",
    "model.encoder.subsample_conv",
    "model.encoder.ca_enrolls.",
    "model.encoder.additional_layer.",
    "model.encoder.additional_self_attention_layer.",
    "model.encoder.scb_layers.",
    "proj_out.",
)


def rename_key(hf_key: str):
    """Rename a single HF key to ESPnet format. Returns None if skipped."""
    if hf_key.startswith(SKIP_PREFIXES):
        return None
    if hf_key == "model.decoder.embed_tokens.weight":
        return None  # handled separately

    new_key = hf_key
    for old_pat, new_pat in RENAME_RULES:
        new_key = new_key.replace(old_pat, new_pat)
    return new_key


def load_hf_state_dict(path: str) -> dict:
    """Load state dict from safetensors or pytorch checkpoint."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        state = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        return state
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint


def convert_state_dict(
    hf_state: dict, n_target_added: int, base_vocab: int = WHISPER_BASE_VOCAB_DEFAULT,
) -> OrderedDict:
    """Convert HF Whisper state dict to ESPnet native Whisper format.

    Args:
        hf_state: HF model state dict.
        n_target_added: number of added tokens expected by the ESPnet model.
        base_vocab: base vocabulary size of the native Whisper model
                    (51865 for most models, 51866 for large-v3-turbo).

    Returns:
        Converted OrderedDict ready for ESPnet model.load_state_dict().
    """
    converted = OrderedDict()
    skipped = []

    for hf_key, tensor in hf_state.items():
        espnet_key = rename_key(hf_key)
        if espnet_key is None:
            skipped.append(hf_key)
            continue
        if espnet_key in converted:
            raise ValueError(
                f"Duplicate ESPnet key '{espnet_key}' from HF key '{hf_key}'"
            )
        converted[espnet_key] = tensor

    # ── Handle token embeddings ──
    hf_emb = hf_state["model.decoder.embed_tokens.weight"]
    v_hf, d_model = hf_emb.shape
    ori_emb = hf_emb[:base_vocab]

    if n_target_added > 0:
        # ESPnet uses ExpandedTokenEmbedding: ori_emb + add_emb
        converted["decoder.decoders.token_embedding.ori_emb.weight"] = ori_emb

        # Initialize add_emb from the distribution of ori_emb
        add_emb = torch.empty(n_target_added, d_model)
        std, mean = torch.std_mean(ori_emb.float())
        torch.nn.init.normal_(add_emb, mean.item(), std.item())

        # Copy matching added token embeddings from source if available
        if v_hf > base_vocab:
            n_source_added = v_hf - base_vocab
            n_copy = min(n_source_added, n_target_added)
            add_emb[:n_copy] = hf_emb[base_vocab : base_vocab + n_copy]
            logging.info(
                f"Copied {n_copy}/{n_target_added} added token embeddings from "
                f"source ({n_source_added} available)"
            )
        else:
            logging.info(
                f"Source has no added tokens; {n_target_added} add_emb rows "
                f"randomly initialized"
            )

        converted["decoder.decoders.token_embedding.add_emb.weight"] = add_emb
    else:
        # No added tokens — standard single embedding
        converted["decoder.decoders.token_embedding.weight"] = ori_emb

    logging.info(f"Converted {len(converted)} keys, skipped {len(skipped)} keys")
    if skipped:
        logging.info("Skipped HF keys:")
        for k in skipped:
            logging.info(f"  {k}")

    return converted


def validate_keys(espnet_state: OrderedDict, config_path: str, token_list_path: str):
    """Build ESPnet model and compare state dict keys.

    Reports missing and unexpected keys. Loads converted weights with
    strict=False to verify shape compatibility.
    """
    import yaml
    from espnet2.tasks.sot_asr import SOTASRTask

    # Build args from config
    parser = SOTASRTask.get_parser()
    args = parser.parse_args(
        ["--config", config_path, "--token_list", token_list_path]
    )
    model = SOTASRTask.build_model(args)

    model_keys = set(model.state_dict().keys())
    converted_keys = set(espnet_state.keys())

    missing = model_keys - converted_keys
    unexpected = converted_keys - model_keys

    if missing:
        logging.info(f"Keys in ESPnet model but not in converted checkpoint ({len(missing)}):")
        for k in sorted(missing):
            logging.info(f"  MISSING: {k}")
    if unexpected:
        logging.warning(
            f"Keys in converted checkpoint but not in ESPnet model ({len(unexpected)}):"
        )
        for k in sorted(unexpected):
            logging.warning(f"  UNEXPECTED: {k}")

    # Try loading
    result = model.load_state_dict(espnet_state, strict=False)
    if result.missing_keys:
        logging.info(
            f"load_state_dict missing_keys ({len(result.missing_keys)}): "
            f"{result.missing_keys[:10]}{'...' if len(result.missing_keys) > 10 else ''}"
        )
    if result.unexpected_keys:
        logging.warning(
            f"load_state_dict unexpected_keys ({len(result.unexpected_keys)}): "
            f"{result.unexpected_keys[:10]}"
        )

    # Check that only expected keys are missing (CTC)
    non_ctc_missing = [k for k in result.missing_keys if not k.startswith("ctc.")]
    if non_ctc_missing:
        logging.error(
            f"Non-CTC keys missing after load — conversion may be incomplete: "
            f"{non_ctc_missing}"
        )
        return False

    logging.info("Validation PASSED: all non-CTC keys loaded successfully")

    # Quick shape sanity check
    for name, param in model.named_parameters():
        if name in espnet_state:
            if param.shape != espnet_state[name].shape:
                logging.error(
                    f"Shape mismatch for {name}: "
                    f"model={param.shape} vs converted={espnet_state[name].shape}"
                )
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Whisper checkpoint to ESPnet SOT format"
    )
    parser.add_argument(
        "--hf_checkpoint",
        type=str,
        required=True,
        help="Path to HF model.safetensors or model.bin file",
    )
    parser.add_argument(
        "--espnet_config",
        type=str,
        required=True,
        help="Path to ESPnet training config YAML (e.g. conf/tuning/train_sot.yaml)",
    )
    parser.add_argument(
        "--token_list",
        type=str,
        required=True,
        help="Path to ESPnet token_list.txt",
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        required=True,
        help="Output path for converted .pth file",
    )
    parser.add_argument(
        "--added_tokens",
        type=str,
        default=None,
        help="Path to added_tokens.txt (overrides token_list for counting added tokens)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Build ESPnet model and validate that converted weights load correctly",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # ── Load HF checkpoint ──
    logging.info(f"Loading HF checkpoint: {args.hf_checkpoint}")
    hf_state = load_hf_state_dict(args.hf_checkpoint)
    logging.info(f"Loaded {len(hf_state)} keys from HF checkpoint")

    # ── Determine base vocab size from config ──
    base_vocab = get_whisper_base_vocab(args.espnet_config)
    logging.info(f"Base Whisper vocabulary size: {base_vocab}")

    # ── Determine number of added tokens in target ──
    # Always compute from token_list and base_vocab — this matches how
    # ESPnet's OpenAIWhisperDecoder splits ori_emb vs add_emb.
    with open(args.token_list, encoding="utf-8") as f:
        token_list = [line.rstrip() for line in f]
    n_target_added = max(0, len(token_list) - base_vocab)

    logging.info(f"Target added tokens: {n_target_added}")

    # ── Convert ──
    espnet_state = convert_state_dict(hf_state, n_target_added, base_vocab=base_vocab)

    # ── Validate ──
    if args.validate:
        logging.info("Running validation...")
        ok = validate_keys(espnet_state, args.espnet_config, args.token_list)
        if not ok:
            logging.error("Validation FAILED — check errors above")
            raise SystemExit(1)

    # ── Save ──
    output_path = Path(args.output_pth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(espnet_state, output_path)
    logging.info(f"Saved converted model to: {output_path}")
    logging.info(f"State dict keys: {len(espnet_state)}")


if __name__ == "__main__":
    main()
