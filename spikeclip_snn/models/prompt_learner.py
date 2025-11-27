"""
Prompt Learner for Stage 2
Following SpikeCLIP paper Section 3.2 and CoOp (Zhou et al. 2022)
https://arxiv.org/abs/2501.04477

Learns separate prompts for HQ and LQ images:
"""

import torch
import torch.nn as nn
import clip

class PromptLearner(nn.Module):
    """
    CoOp-style learnable prompt for HQ/LQ classification
    """

    def __init__(
        self,
        clip_model,
        n_ctx=4,
        ctx_init="",
        class_token_position="end"
    ):
        """
        :param clip_model: Pretrained CLIP model
        :param n_ctx: Number of learnable context tokens
        :param ctx_init: Optional initialization text for context
        :param class_token_position: Where to put class token ("end", "middle", "front")
        """
        super().__init__()

        self.n_ctx = n_ctx
        self.class_token_position = class_token_position

        # Get CLIP text encoder properties
        # 512 for ViT-B/32
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Store to use it later
        self.dtype = dtype
        self.ctx_dim = ctx_dim

        # Get device from CLIP model
        device = clip_model.token_embedding.weight.device

        # Initialize learnable context vectors
        if ctx_init and len(ctx_init.strip()) > 0:
            # Initialize from text
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1+n_ctx, :]
            self.n_ctx = n_ctx

        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)

        # Learnable context
        # [V1][V2][V3][V4] tokens in our case
        self.ctx = nn.Parameter(ctx_vectors)

        # Fixed class names for HQ/LQ
        self.classnames = ["low quality", "high quality"]
        self.n_cls = len(self.classnames)

        # Create template prompts with placeholder "X" for learnable tokens
        # Template: "X X X X {classname}."
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in self.classnames]

        # Tokenize class names
        tokenized_prompts = clip.tokenize(prompts).to(device)

        # Get the token embedding for class names
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Register buffers (non-learnable but should move with model)
        # SOS token
        # class + EOS
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # Health Check
        print(f"PromptLearner initialized:")
        print(f"  Learnable context tokens: {n_ctx}")
        print(f"  Context dimension: {ctx_dim}")
        print(f"  Classes: {self.classnames}")

    def forward(self):
        """
        Construct the full prompt embeddings
        :return: prompt embeddings (n_cls, 77, ctx_dim)
        """
        # (n_ctx, ctx_dim)
        ctx = self.ctx

        # Expand context for each class (n_cls, n_ctx, ctx_dim)
        # IMPORTANT: Use .repeat() not .expand() to allow gradient flow
        ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)

        # (n_cls, 1, ctx_dim) - SOS
        # (n_cls, *, ctx_dim) - class name + EOS + padding
        prefix = self.token_prefix
        suffix = self.token_suffix

        # Concatenate, [SOS] [V1][V2][V3][V4] [class name] [EOS] [PAD...]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class TextEncoder(nn.Module):
    """
    Wrapper around CLIP's text encoder
    Takes prompt embeddings and returns text features.

    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Encode prompts to text features.

        :param prompts: Prompt embeddings from PromptLearner (n_cls, 77, ctx_dim)
        :param tokenized_prompts: Original tokenized prompts for EOS position
        :return: Text features (n_cls, 512)
        """
        # Add positional embeddings
        x = prompts + self.positional_embedding.type(self.dtype)

        # Transformer
        # (77, n_cls, ctx_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # (n_cls, 77, ctx_dim)
        x = x.permute(1, 0, 2)

        # Layer norm
        x = self.ln_final(x).type(self.dtype)

        # Get features at EOS token position
        # EOS token ID in CLIP is 49407
        eos_token_id = 49407
        device = x.device

        # Find EOS position for each prompt
        eos_mask = (tokenized_prompts == eos_token_id)
        eos_positions = eos_mask.nonzero(as_tuple=True)[1]

        # Fallback if EOS not found
        if len(eos_positions) != x.shape[0]:
            eos_positions = (tokenized_prompts != 0).sum(dim=1) - 1

        # Extract features at EOS positions
        x = x[torch.arange(x.shape[0], device=device), eos_positions]

        # Project to CLIP embedding space
        x = x @ self.text_projection

        return x


class PromptCLIP(nn.Module):
    """
    Full model combining PromptLearner + TextEncoder + CLIP ImageEncoder
    Used for Stage 2 training
    """

    def __init__(self, clip_model, n_ctx=4):
        """
        :param clip_model: Pretrained CLIP model (frozen)
        :param n_ctx: Number of learnable context tokens
        """
        super().__init__()

        self.prompt_learner = PromptLearner(clip_model, n_ctx=n_ctx)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Freeze text encoder (only prompt_learner.ctx is trainable)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\nPromptCLIP initialized:")
        print(f"  Trainable parameters: {trainable:,} (only prompt tokens!)")
        print(f"  Total parameters: {total:,}")

    def forward(self, images):
        """
        Forward pass for HQ/LQ classification.

        :param images: Batch of images (B, 3, 224, 224)
        :return: logits (B, 2) - probability of [LQ, HQ]
        """
        # Get image features
        image_features = self.image_encoder(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get text features from learnable prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_prompt_features(self):
        """
        Get the current text features for HQ and LQ prompts.
        Useful for visualization and debugging.

        :return: text_features (2, 512)
        """
        with torch.no_grad():
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features










































