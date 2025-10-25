import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig


def ParallelHFModel(model_hf_tag, **kwargs):
    model_class = build_parallel_hf_class(model_hf_tag)
    return model_class.from_pretrained(model_hf_tag, **kwargs)


def build_parallel_hf_class(model_hf_tag):

    config = AutoConfig.from_pretrained(model_hf_tag)
    architecture = config.architectures[0]
    architecture = getattr(transformers, architecture)

    class ParallelLLM(architecture):
        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            multimodal_io,
            vocab_intervals,
            max_loss_interval: int = 13192,
            **kwargs,
        ):
            # (1) Load the base model using parent's from_pretrained
            model = super(ParallelLLM, cls).from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

            # (2) rebuild the input/output embedding tables.
            # (a) place 0 as all-zero embedding
            # (b) replace text embeddings from the pre-trained weights.
            with torch.no_grad():
                vocab_size = max(
                    [
                        end
                        for intervals in vocab_intervals.values()
                        for _, end in intervals
                    ]
                )

                embed_dim = model.config.hidden_size
                new_embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
                new_embed_tokens.weight[0] = 0.0
                new_lm_head.weight[0] = 0.0

                if "text" in vocab_intervals:

                    if not (
                        hasattr(model, "model") and hasattr(model.model, "embed_tokens")
                    ):
                        raise AttributeError(
                            "Model must have 'model.embed_tokens' attribute"
                        )
                    if not hasattr(model, "lm_head"):
                        raise AttributeError("Model must have 'lm_head' attribute")

                    text_start, text_end = vocab_intervals["text"][0]

                    old_embed = model.model.embed_tokens
                    old_lm_head = model.lm_head
                    orig_vocab_size = old_embed.weight.shape[0]

                    if text_end - text_start != orig_vocab_size:
                        raise ValueError(
                            f"text_end - text_start ({text_end - text_start}) must equal "
                            f"original vocab size ({orig_vocab_size})"
                        )

                    embed_dim = model.config.hidden_size
                    new_embed_tokens = nn.Embedding(vocab_size, embed_dim)
                    new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

                    new_embed_tokens.weight[text_start:text_end] = old_embed.weight
                    new_lm_head.weight[text_start:text_end] = old_lm_head.weight

                model.model.embed_tokens = new_embed_tokens
                model.lm_head = new_lm_head

            # (3) build stream embeddings. First stream doesn't apply this embedding
            possible_num_stream = [
                io.num_stream() for io in multimodal_io.values() if io.is_discrete
            ]
            if len(possible_num_stream) == 0:
                raise ValueError("Cannot proceed with all IOs being continuous")
            model.num_stream = max(possible_num_stream)
            model.stream_emb = nn.Embedding(model.num_stream, embed_dim)

            # (4) multimodal IO
            model.multimodal_io_dict = nn.ModuleDict(multimodal_io)
            model.adaptor = nn.ModuleDict()
            for io_name, io in model.multimodal_io_dict.items():
                if not io.is_discrete:
                    model.adaptor[io_name] = nn.Linear(
                        io.feature_dim(),
                        model.config.hidden_size,
                    )

            # (5) loss computing interval
            model.vocab_intervals = vocab_intervals
            model.loss_intervals = list()
            for io_name, intervals in vocab_intervals.items():
                if io_name == "text" or io_name == "special_token":
                    continue

                cur_start, _ = intervals[0]
                for _, end in intervals[1:]:
                    if end - cur_start <= max_loss_interval:
                        continue
                    else:
                        model.loss_intervals.append((cur_start, end))
                        cur_start = end

                if end > cur_start:
                    model.loss_intervals.append((cur_start, end))

            return model

        def forward(self, **kwargs):
            """Forward without loss computing"""
            input_ids = kwargs["seqs"]
            conti_feats = kwargs["conti_feats"]
            loss_mask = kwargs.get("loss_masks", None)
            position_ids = kwargs.get("position_ids", None)
            past_key_values = kwargs.get("past_key_values", None)

            inputs_embeds = self._embed(input_ids, conti_feats)

            # Forward on base model
            output = self.model.forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            )

            hidden_states = output.last_hidden_state.unsqueeze(2)
            stream_emb = self.stream_emb.weight.tile(1, 1, 1, 1)
            stream_emb[:, :, 0] = 0.0
            hidden_states = hidden_states + stream_emb

            if loss_mask is not None:
                loss, stats = self._loss(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                )
                return {"loss": loss, "stats": stats}

            else:
                logits = self.lm_head(hidden_states)
                return {"logits": logits}

        def _embed(self, input_ids, conti_feats):

            # (1) On-the-fly discrete tokens
            for io_name, (io_index, io_feats) in conti_feats.items():
                if not self.multimodal_io_dict[io_name].is_discrete:
                    continue
                codes = self.multimodal_io_dict[io_name].encode_batch(*io_feats)
                codes = codes + self.vocab_intervals[io_name][0][0]
                for code, (bidx, start, length) in zip(codes, io_index):
                    input_ids[bidx, start : start + length] = code[:length]

            # (2) embeddings
            input_embeds = self.model.embed_tokens(input_ids).sum(dim=2)

            # (3) On-the-fly continuous features
            for io_name, (io_index, io_feats) in conti_feats.items():
                if self.multimodal_io_dict[io_name].is_discrete:
                    continue
                io_feats = self.multimodal_io_dict[io_name].encode_batch(*io_feats)
                for feat, (bidx, start, length) in zip(io_feats, io_index):
                    feat = self.adaptor[io_name](feat)
                    input_embeds[bidx, start : start + length] = feat[:length]

            return input_embeds

        def _loss(self, hidden_states, input_ids, loss_mask):
            assert input_ids.size() == loss_mask.size()
            assert hidden_states.size()[:3] == loss_mask.size()

            hidden_states = hidden_states[:, :-1]
            input_ids = input_ids[:, 1:]
            loss_mask = loss_mask[:, 1:]

            loss = torch.zeros_like(loss_mask)
            acc = torch.zeros_like(loss_mask).bool()
            stats = dict()

            # Full softmax for the first stream
            this_mask = torch.zeros_like(input_ids).bool()
            this_mask[:, :, 0] = True

            this_logits = hidden_states[this_mask]
            this_logits = torch.matmul(this_logits, self.lm_head.weight.T)
            this_targets = input_ids[this_mask]

            this_loss = torch.nn.functional.cross_entropy(
                this_logits,
                this_targets,
                reduction="none",
                ignore_index=0,
            )
            loss.masked_scatter_(this_mask, this_loss)
            if not self.training:
                this_acc = this_logits.argmax(-1) == this_targets
                acc.masked_scatter_(this_mask, this_acc)

            # interval softmax for the rest stream
            for start, end in self.loss_intervals:
                this_mask = torch.logical_and(input_ids >= start, input_ids < end)
                if this_mask.int().sum() == 0:
                    continue
                this_logits = hidden_states[this_mask]
                this_logits = torch.matmul(
                    this_logits, self.lm_head.weight[start:end].T
                )
                this_targets = input_ids[this_mask] - start
                this_loss = torch.nn.functional.cross_entropy(
                    this_logits,
                    this_targets,
                    reduction="none",
                )
                loss.masked_scatter_(this_mask, this_loss)
                if not self.training:
                    this_acc = this_logits.argmax(-1) == this_targets - start
                    acc.masked_scatter_(this_mask, this_acc)

            loss = loss * loss_mask
            count = (loss_mask != 0.0).float()
            loss = loss.sum() / count.sum()
            stats["loss"] = loss.clone().detach()

            if not self.training:
                acc = acc.float()
                stats["acc"] = acc.sum() / count.sum()
                for n in range(self.num_stream):
                    this_count = count[:, :, n].sum()
                    if this_count > 0:
                        stats[f"acc_layer{n}"] = acc[:, :, n].sum() / this_count

            return loss, stats

    return ParallelLLM
