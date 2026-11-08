"""
Unified interface for length-controlled generation with diffusion LLMs.
Supports: LLaDA-8B, Dream-7B
"""

import torch
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Literal
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


class DiffusionLM(ABC):
    """Abstract base class for diffusion language models."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._token_lengths = None

    @property
    @abstractmethod
    def mask_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        pass

    @property
    @abstractmethod
    def eot_id(self) -> int:
        pass

    @property
    def vocab_size(self) -> int:
        return self.model.get_input_embeddings().weight.shape[0]

    @property
    def token_lengths(self) -> torch.Tensor:
        if self._token_lengths is None:
            self._token_lengths = self._build_token_length_vector()
        return self._token_lengths

    def _build_token_length_vector(self) -> torch.Tensor:
        lengths = torch.zeros(self.vocab_size, device=self.device)
        special_ids = {
            self.mask_id,
            self.pad_id,
            self.eot_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        }

        for token_id in range(self.vocab_size):
            if token_id in special_ids:
                lengths[token_id] = 0
            else:
                try:
                    text = self.tokenizer.decode([token_id])
                    lengths[token_id] = len(text)
                except Exception:
                    lengths[token_id] = 0
        return lengths

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for input_ids."""
        pass

    @abstractmethod
    def get_output_logits(self, logits: torch.Tensor, gen_start: int) -> torch.Tensor:
        """Extract output logits from full logits tensor.

        Different models have different output alignments:
        - LLaDA: logits[gen_start:] corresponds to positions gen_start:
        - Dream: logits are shifted, so logits[gen_start-1:-1] corresponds to positions gen_start:
        """
        pass

    def encode_prompt(self, text: str, use_chat_template: bool = True) -> torch.Tensor:
        """Encode text to input_ids, optionally using chat template for instruct models."""
        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
            return input_ids
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token_ids to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_input_embeddings(self):
        """Get input embedding layer."""
        return self.model.get_input_embeddings()


class LLaDAModel(DiffusionLM):
    """LLaDA-8B wrapper."""

    MASK_ID = 126336
    PAD_ID = 126081
    EOT_ID = 126348

    @property
    def mask_id(self) -> int:
        return self.MASK_ID

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def eot_id(self) -> int:
        return self.EOT_ID

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    def get_output_logits(self, logits: torch.Tensor, gen_start: int) -> torch.Tensor:
        """LLaDA: logits are aligned, position i predicts token i."""
        return logits[0, gen_start:]


class DreamModel(DiffusionLM):
    """Dream-7B wrapper."""

    @property
    def mask_id(self) -> int:
        return self.tokenizer.mask_token_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    @property
    def eot_id(self) -> int:
        return self.tokenizer.eos_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    def get_output_logits(self, logits: torch.Tensor, gen_start: int) -> torch.Tensor:
        """Dream: logits are shifted by 1, position i-1 predicts token i."""
        return logits[0, gen_start - 1:-1]


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    quantization: Optional[Literal["4bit", "8bit"]] = None,
) -> DiffusionLM:
    """Load a diffusion LM by model ID.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Data type for model weights, ignored if quantization is set.
        quantization: Optional quantization mode, either "4bit", "8bit", or None.

    Returns:
        DiffusionLM instance.
    """
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = quantization_config
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(
        model_id,
        **load_kwargs,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    device = model.device

    if "dream" in model_id.lower():
        return DreamModel(model, tokenizer, device)
    elif "llada" in model_id.lower():
        return LLaDAModel(model, tokenizer, device)
    else:
        raise ValueError(f"Unknown model type: {model_id}")


@torch.no_grad()
def margin_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence scores based on margin likelihood.

    Margin = max probability - second highest probability.
    Higher margin indicates more confident predictions.

    Args:
        probs: Probability distribution over vocabulary, shape (L, V)

    Returns:
        Margin confidence scores in [0, 1], shape (L,)
    """
    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    return margin


@torch.no_grad()
def entropy_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence scores based on entropy.

    Low entropy = high confidence. Scaled to [0, 1] where 1 = perfectly confident.
    """
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    max_entropy = math.log(probs.shape[-1])
    confidence = 1.0 - (entropy / max_entropy)

    return confidence.clamp(0, 1)


@torch.no_grad()
def probability_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Compute confidence scores based on max probability."""
    return probs.max(dim=-1).values


@torch.no_grad()
def semantic_confidence(probs: torch.Tensor, embedding_weights: torch.Tensor) -> torch.Tensor:
    """Compute confidence scores based on semantic clustering in embedding space."""
    probs_f32 = probs.float()
    embed_f32 = embedding_weights.float()

    expected_emb = probs_f32 @ embed_f32
    expected_emb_norm = F.normalize(expected_emb, p=2, dim=-1)

    embed_norm = F.normalize(embed_f32, p=2, dim=-1)
    cos_sim = expected_emb_norm @ embed_norm.T

    confidence = (probs_f32 * cos_sim).sum(dim=-1)
    confidence = (confidence + 1) / 2

    return confidence.clamp(0, 1)


@torch.no_grad()
def semantic_variance_confidence(probs: torch.Tensor, embedding_weights: torch.Tensor) -> torch.Tensor:
    """Compute confidence scores based on variance in embedding space."""
    probs_f32 = probs.float()
    embed_f32 = embedding_weights.float()

    expected_emb = probs_f32 @ embed_f32

    embed_sq_norms = (embed_f32 ** 2).sum(dim=-1)
    expected_sq_norms = (expected_emb ** 2).sum(dim=-1)
    dot_products = expected_emb @ embed_f32.T

    sq_distances = embed_sq_norms.unsqueeze(0) - 2 * dot_products + expected_sq_norms.unsqueeze(1)
    variance = (probs_f32 * sq_distances).sum(dim=-1)

    max_variance = 4 * embed_sq_norms.max()
    confidence = 1.0 - (variance / max_variance).clamp(0, 1)

    return confidence.clamp(0, 1)


def compute_confidence(
    probs: torch.Tensor,
    metric: str = "margin",
    embedding_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute confidence scores using the specified metric.

    Args:
        probs: Probability distribution over vocabulary, shape (L, V)
        metric: One of "margin", "entropy", "probability", "semantic", "semantic_variance"
        embedding_weights: Token embedding matrix, required for semantic metrics

    Returns:
        Confidence scores in [0, 1], shape (L,)
    """
    if metric == "margin":
        return margin_confidence(probs)
    elif metric == "entropy":
        return entropy_confidence(probs)
    elif metric == "probability":
        return probability_confidence(probs)
    elif metric == "semantic":
        if embedding_weights is None:
            raise ValueError("embedding_weights required for semantic confidence")
        return semantic_confidence(probs, embedding_weights)
    elif metric == "semantic_variance":
        if embedding_weights is None:
            raise ValueError("embedding_weights required for semantic_variance confidence")
        return semantic_variance_confidence(probs, embedding_weights)
    else:
        raise ValueError(
            f"Unknown confidence metric: {metric}. "
            "Choose from: margin, entropy, probability, semantic, semantic_variance"
        )


def estimate_initial_buffer_size(
    target_char_length: int,
    initial_gen_length: int,
    buffer_size_mode: str,
    avg_chars_per_token: Optional[float] = None,
) -> int:
    """
    Estimate the number of masked token positions used to initialize generation.

    avg_chars_per_token: round(target_char_length / avg_chars_per_token)
    manual: use initial_gen_length directly
    target_over_2: round(target_char_length / 2)
    """
    if buffer_size_mode == "avg_chars_per_token":
        if avg_chars_per_token is None or avg_chars_per_token <= 0:
            raise ValueError(
                "buffer_size_mode='avg_chars_per_token' requires avg_chars_per_token > 0"
            )
        gen_length = int(round(target_char_length / avg_chars_per_token))
    elif buffer_size_mode == "manual":
        gen_length = int(initial_gen_length)
    elif buffer_size_mode == "target_over_2":
        gen_length = int(round(target_char_length / 2))
    else:
        raise ValueError(
            f"Unknown buffer_size_mode={buffer_size_mode}. "
            "Choose from: avg_chars_per_token, manual, target_over_2"
        )

    return max(4, gen_length)


@torch.no_grad()
def generate_length_controlled(
    dlm: DiffusionLM,
    prompt: torch.Tensor,
    target_char_length: int,
    initial_gen_length: int = 128,
    max_iterations: int = 100,
    confidence_threshold: float = 1.0,
    remask_threshold: float = 0.0,
    low_confidence_threshold: float = 1.0,
    confidence_metric: str = "margin",
    enable_length_adjustment: bool = True,
    avg_chars_per_token: Optional[float] = None,
    buffer_size_mode: str = "avg_chars_per_token",
    debug: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """
    Generate text with target character length using iterative diffusion.

    Args:
        dlm: DiffusionLM instance
        prompt: Prompt token IDs [1, prompt_len]
        target_char_length: Target output length in characters
        initial_gen_length: Manual initial generation buffer size
        max_iterations: Maximum diffusion iterations
        confidence_threshold: Threshold for unmasking tokens
        remask_threshold: Threshold below which fixed tokens get remasked, 0 to disable
        low_confidence_threshold: Threshold below which mask tokens can be removed for length control
        confidence_metric: One of "margin", "entropy", "probability", "semantic", "semantic_variance"
        enable_length_adjustment: If False, disables insert/remove for fixed-buffer diagnostic
        avg_chars_per_token: Validation-estimated chars/token ratio for avg_chars_per_token mode
        buffer_size_mode: One of "avg_chars_per_token", "manual", "target_over_2"
        debug: Print debug info

    Returns:
        (output_ids, num_iterations, num_remasks)
    """
    device = dlm.device
    gen_start = prompt.shape[1]
    token_lengths = dlm.token_lengths

    gen_length = estimate_initial_buffer_size(
        target_char_length=target_char_length,
        initial_gen_length=initial_gen_length,
        buffer_size_mode=buffer_size_mode,
        avg_chars_per_token=avg_chars_per_token,
    )

    if max_iterations is None or max_iterations <= 0:
        max_iterations = gen_length * (2 if enable_length_adjustment else 1)

    embedding_weights = None
    if confidence_metric in ("semantic", "semantic_variance"):
        embedding_weights = dlm.get_input_embeddings().weight

    if debug:
        print(
            f"\n=== Starting with gen_length={gen_length}, metric={confidence_metric}, "
            f"buffer_size_mode={buffer_size_mode}, avg_chars_per_token={avg_chars_per_token}, "
            f"length_adjustment={enable_length_adjustment} ==="
        )

    x = torch.full(
        (1, gen_start + gen_length),
        dlm.mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :gen_start] = prompt.clone()
    x[:, -1] = dlm.eot_id

    last_action = "unmask"
    num_remasks = 0

    for iteration in range(max_iterations):
        output_tokens = x[0, gen_start:]
        mask_positions = output_tokens == dlm.mask_id
        n_masks = mask_positions.sum().item()
        n_tokens = len(output_tokens)

        logits = dlm.forward(x)
        output_logits = dlm.get_output_logits(logits, gen_start)

        empty_token_mask = token_lengths == 0
        masked_logits = output_logits.clone()
        masked_logits[:, empty_token_mask] = float("-inf")

        probs = F.softmax(masked_logits, dim=-1)
        predictions = masked_logits.argmax(dim=-1)

        confidence = compute_confidence(probs, confidence_metric, embedding_weights)
        max_prob = compute_confidence(probs, "probability", embedding_weights)

        if remask_threshold > 0:
            fixed_positions = ~mask_positions
            low_margin_fixed = fixed_positions & (confidence < remask_threshold)
            n_remask = low_margin_fixed.sum().item()

            if n_remask > 0:
                if debug:
                    remasked_tokens = output_tokens[low_margin_fixed]
                    remasked_text = dlm.decode(remasked_tokens)
                    print(f"    -> REMASKING {n_remask} low-margin tokens: '{remasked_text}'")

                output_tokens = output_tokens.clone()
                output_tokens[low_margin_fixed] = dlm.mask_id
                x = torch.cat([x[:, :gen_start], output_tokens.unsqueeze(0)], dim=1)
                num_remasks += n_remask

                mask_positions = output_tokens == dlm.mask_id
                n_masks = mask_positions.sum().item()

        expected_lens_per_token = torch.zeros(n_tokens, device=device)

        unmasked_indices = (~mask_positions).nonzero(as_tuple=True)[0]
        if len(unmasked_indices) > 0:
            unmasked_tokens = output_tokens[unmasked_indices]
            expected_lens_per_token[unmasked_indices] = token_lengths[unmasked_tokens].float()

        mask_indices = mask_positions.nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            expected_lens_per_token[mask_indices] = (probs[mask_indices] * token_lengths).sum(dim=-1)

        expected_len = expected_lens_per_token.sum().item()
        mean_token_len = expected_lens_per_token.mean().item() if n_tokens > 0 else 4.0

        if debug:
            n_unmasked = n_tokens - n_masks
            print(
                f"  iter={iteration}: tokens={n_tokens} (unmasked={n_unmasked}, masks={n_masks}), "
                f"expected={expected_len:.0f}, target={target_char_length}"
            )

        if n_masks == 0:
            if debug:
                print(f"  Total remasks during generation: {num_remasks}")
            return x, iteration + 1, num_remasks

        if enable_length_adjustment:
            del_scores = 1 - confidence
            mask_del_scores = del_scores[mask_indices]
            sorted_order = torch.argsort(mask_del_scores, descending=True)
            sorted_mask_indices = mask_indices[sorted_order]

            cumsum = torch.cumsum(expected_lens_per_token[sorted_mask_indices], dim=0)
            cumsum = torch.cat([torch.tensor([0.0], device=device), cumsum], dim=0)

            low_confidence_mask = confidence[mask_indices] < low_confidence_threshold
            n_remove = min(
                low_confidence_mask.sum().item(),
                abs(expected_len - cumsum - target_char_length).argmin().item(),
            )
            n_add = int((target_char_length - expected_len) / mean_token_len) if mean_token_len > 1e-6 else 0

            if n_tokens > n_remove > 0 and last_action == "unmask":
                last_action = "remove"
                remove_idx = sorted_mask_indices[:n_remove]
                keep_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
                keep_mask[remove_idx] = False
                removed_len = expected_lens_per_token[sorted_mask_indices][:n_remove].sum().item()
                if debug:
                    print(f"    -> removing {n_remove} masks (expected {removed_len:.0f} chars)")

                output_tokens = output_tokens[keep_mask]
                x = torch.cat([x[:, :gen_start], output_tokens.unsqueeze(0)], dim=1)
                continue

            elif n_add > 0 and last_action == "unmask":
                last_action = "insert"
                insert_idx = sorted_mask_indices[:n_add]
                new_output_tokens = torch.full((n_tokens + n_add,), dlm.mask_id, dtype=torch.long, device=device)
                insert_offsets = torch.zeros(n_tokens, dtype=torch.long, device=device)
                insert_offsets[insert_idx] = 1
                insert_offsets = torch.arange(n_tokens, device=device) + torch.cumsum(insert_offsets, dim=0)
                new_output_tokens[insert_offsets] = output_tokens
                x = torch.cat([x[:, :gen_start], new_output_tokens.unsqueeze(0)], dim=1)

                if debug:
                    print(f"    -> inserting {n_add} masks")
                continue

        valid_mask_positions = (output_tokens == dlm.mask_id) & (token_lengths[predictions] > 0)
        unmask = valid_mask_positions & (max_prob > confidence_threshold)
        mask_confidence = torch.where(
            valid_mask_positions,
            max_prob,
            torch.tensor(-float("inf"), device=device),
        )
        best_mask_idx = mask_confidence.argmax()
        unmask[best_mask_idx] = True

        if debug:
            print(f"    -> unmasking {unmask.sum().item()} high-margin tokens")
            print(dlm.decode(predictions[unmask]))

        output_tokens = output_tokens.clone()
        output_tokens[unmask] = predictions[unmask]
        x = torch.cat([x[:, :gen_start], output_tokens.unsqueeze(0)], dim=1)
        last_action = "unmask"

    if debug:
        print(f"  Total remasks during generation: {num_remasks}")
    return x, max_iterations, num_remasks
