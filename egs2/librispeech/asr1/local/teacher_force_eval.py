#!/usr/bin/env python3
"""Teacher-forced evaluation: feed ground truth tokens to the decoder,
get the model's per-position argmax predictions, and write output in
the same text format as autoregressive inference for comparison.

Usage:
    python local/teacher_force_eval.py \
        --config $EXP_DIR/config.yaml \
        --checkpoint $EXP_DIR/51epoch.pth \
        --data-dir dump/raw/maestro_train_segments \
        --output teacher_forced_text
"""

import argparse

import numpy as np
import soundfile as sf
import torch
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    parser.add_argument("--data-dir", required=True, help="Directory with wav.scp and text")
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device

    # 1. Load model
    print("Loading model...")
    asr_model, asr_train_args = ASRTask.build_model_from_file(
        args.config, args.checkpoint, device=device
    )
    asr_model.eval()

    # 2. Build tokenizer and token ID converter (word type = whitespace split)
    token_list = asr_train_args.token_list
    tokenizer = build_tokenizer(token_type="word")
    converter = TokenIDConverter(token_list=token_list)

    sos = asr_model.sos
    eos = asr_model.eos
    ignore_id = asr_model.ignore_id

    # 3. Load data
    wav_scp = {}
    with open(f"{args.data_dir}/wav.scp") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            wav_scp[parts[0]] = parts[1]

    ref_text = {}
    with open(f"{args.data_dir}/text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            ref_text[parts[0]] = parts[1] if len(parts) > 1 else ""

    # 4. Run teacher-forced inference
    results = []

    with torch.no_grad():
        for utt_id in sorted(wav_scp.keys()):
            # Load audio
            speech_np, sr = sf.read(wav_scp[utt_id])
            assert sr == 16000
            speech = torch.tensor(speech_np, dtype=torch.float32).unsqueeze(0).to(device)
            speech_lengths = torch.tensor([speech.size(1)], dtype=torch.long).to(device)

            # Encode
            encoder_out, encoder_out_lens = asr_model.encode(speech, speech_lengths)

            # Tokenize ground truth text
            text_str = ref_text.get(utt_id, "")
            tokens = tokenizer.text2tokens(text_str)
            token_ids = converter.tokens2ids(tokens)
            text_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
            text_lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(device)

            # Add SOS/EOS
            ys_in_pad, ys_out_pad = add_sos_eos(text_tensor, sos, eos, ignore_id)
            ys_in_lens = text_lengths + 1

            # Decoder forward (teacher forcing)
            decoder_out, _ = asr_model.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )
            # decoder_out: (1, SeqLen+1, VocabSize)

            # Argmax predictions
            pred_ids = decoder_out.argmax(dim=-1)[0]  # (SeqLen+1,)

            # ys_out_pad is [tok1, tok2, ..., tokN, EOS, pad...]
            # pred_ids aligns with ys_out_pad: position i predicts ys_out_pad[i]
            # We want predictions for positions 0..N-1 (the actual tokens, not EOS)
            n_tokens = len(token_ids)
            pred_token_ids = pred_ids[:n_tokens].cpu().tolist()

            # Convert back to strings
            pred_tokens = converter.ids2tokens(pred_token_ids)
            pred_str = " ".join(pred_tokens)

            # Compute per-token accuracy
            ref_ids = token_ids
            n_correct = sum(1 for p, r in zip(pred_token_ids, ref_ids) if p == r)
            acc = n_correct / len(ref_ids) * 100 if ref_ids else 0

            print(f"  {utt_id}: {len(ref_ids)} tokens, acc={acc:.1f}%")
            results.append(f"{utt_id} {pred_str}")

    # 5. Write output
    with open(args.output, "w") as f:
        f.write("\n".join(results) + "\n")

    print(f"\nDone! Wrote {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
