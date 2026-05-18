"""Small helper functions shared by the LTX-Video POWER8 scripts."""


def pack_latents(latents, patch_size=1, patch_size_t=1):
    """Pack latents from [B, C, F, H, W] to [B, num_tokens, C]."""
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size

    latents = latents.reshape(
        batch_size,
        num_channels,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    latents = latents.reshape(
        batch_size,
        post_patch_num_frames * post_patch_height * post_patch_width,
        num_channels * patch_size_t * patch_size * patch_size,
    )
    return latents


def unpack_latents(
    latents,
    num_frames,
    height,
    width,
    patch_size=1,
    patch_size_t=1,
    out_channels=128,
):
    """Unpack latents from [B, num_tokens, C] back to [B, C, F, H, W]."""
    batch_size = latents.shape[0]
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size

    latents = latents.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        out_channels,
        patch_size_t,
        patch_size,
        patch_size,
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = latents.reshape(
        batch_size,
        out_channels,
        num_frames,
        height,
        width,
    )
    return latents


def map_key(key):
    """Map 13B checkpoint keys to diffusers model keys."""
    mapped = key.replace("model.diffusion_model.", "")
    mappings = [
        ("patchify_proj.", "proj_in."),
        (".q_norm.", ".norm_q."),
        (".k_norm.", ".norm_k."),
        ("adaln_single.emb.timestep_embedder.", "time_embed.emb.timestep_embedder."),
        ("adaln_single.linear.", "time_embed.linear."),
    ]
    for old, new in mappings:
        mapped = mapped.replace(old, new)
    return mapped
