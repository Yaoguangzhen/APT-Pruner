import torch


def mac_per_head(seq_len, hidden_size, attention_head_size):
    """Calculate MACs for a single attention head."""
    # QKV projection: 3 * (N * D * d_head)
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    # Attention Score (QK^T) and Weighted Sum (AV): 2 * (N^2 * d_head)
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    # Output projection: N * d_head * D
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size

    if seq_len <= 0:
        return 0
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    """Calculate MACs for a single FFN neuron (intermediate dimension unit)."""
    # FFN layers: (N * D * 1) + (N * 1 * D) = 2 * N * D
    if seq_len <= 0:
        return 0
    return 2 * seq_len * hidden_size


def mac_per_token(seq_len, hidden_size, intermediate_size, num_attention_heads):
    """Calculate the approximate MACs saved by removing one token per layer.

    Args:
        seq_len (int): Original sequence length.
        hidden_size (int): Hidden dimension size.
        intermediate_size (int): Size of the FFN intermediate layer.
        num_attention_heads (int): Number of attention heads.

    Returns:
        float: The approximate MACs saved per layer by removing one token.
    """
    if seq_len <= 1:
        # Removing the last token removes all computation associated with it
        seq_len = 1

    attention_head_size = int(hidden_size / num_attention_heads)

    # MACs for original sequence length (L)
    original_layer_macs = num_attention_heads * mac_per_head(seq_len, hidden_size, attention_head_size) + \
                          intermediate_size * mac_per_neuron(seq_len, hidden_size)

    # MACs for sequence length (L-1)
    pruned_layer_macs = num_attention_heads * mac_per_head(seq_len - 1, hidden_size, attention_head_size) + \
                        intermediate_size * mac_per_neuron(seq_len - 1, hidden_size)

    mac_saved_per_token_per_layer = original_layer_macs - pruned_layer_macs

    return max(0.0, mac_saved_per_token_per_layer)


def compute_mac(num_heads_per_layer, num_neurons_per_layer, seq_len, hidden_size, attention_head_size):
    """Compute total MACs for the model given specific configurations."""
    mac = 0.0
    # Handle scalar inputs by repeating them if necessary, or assume they are lists
    if not isinstance(num_heads_per_layer, (list, tuple)):
        # Assuming uniform layers if not list (though typical usage passes lists)
        # This part depends on how it's called. Keeping original logic structure.
        pass

    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(seq_len, hidden_size, attention_head_size)
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size)
        mac += attention_mac + ffn_mac
    return mac


def compute_token_pruned_mac(config, seq_len, num_tokens_kept):
    """Computes the MAC count before and after token pruning."""
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    # Original MAC (with full sequence length)
    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )

    # Pruned MAC (with reduced sequence length)
    # Token pruning reduces seq_len, not the number of heads or neurons
    pruned_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        num_tokens_kept,
        hidden_size,
        attention_head_size,
    )
    return pruned_mac, original_mac


def prune_tokens_vit(model, token_mask, mha_lut=None, ffn_lut=None):
    """
    Simulates token pruning forward pass (masking) and calculates MAC reduction.
    Note: Real speedup requires 'gather' operations, this function calculates theoretical MACs.
    """
    # 1. Handle Embedding Layer variations
    if hasattr(model, 'vit') and hasattr(model.vit, 'embeddings'):
        input_embeddings = model.vit.embeddings
        # For HuggingFace ViT, we typically mask after embeddings or use pixel_values
        # This function signature assumes we are manipulating embeddings directly or hooks.
        # Placeholder for logic if we needed to run forward pass:
        device = input_embeddings.patch_embeddings.projection.weight.device
    elif hasattr(model, 'embed_tokens'):
        input_embeddings = model.embed_tokens
        device = input_embeddings.weight.device
    else:
        # Fallback or dummy
        device = token_mask.device

    token_mask = token_mask.to(device)

    # 2. Compute MACs based on Kept Tokens
    # We calculate how many tokens are active (mask value > 0)
    # Assuming token_mask is 1D [seq_len] or 2D [batch, seq_len]
    if token_mask.dim() == 2:
        num_tokens_kept = token_mask[0].sum().item()  # Assume batch size 1 or uniform
    else:
        num_tokens_kept = token_mask.sum().item()

    # Use the correct function for token pruning MACs
    pruned_mac, original_mac = compute_token_pruned_mac(
        model.config if hasattr(model, 'config') else model,  # Handle dummy model
        model.seq_len if hasattr(model, 'seq_len') else token_mask.shape[-1],  # inferred seq_len
        num_tokens_kept
    )

    # 3. (Optional) Forward pass with masking for verification
    # Note: To actually prune in forward, usually requires modifying input or hooks.
    # Returning None for output as this is 'efficiency/mac.py', primarily for stats.
    model_output = None

    return model_output, pruned_mac, original_mac


if __name__ == "__main__":
    # Test Case
    token_mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)


    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
            self.hidden_size = 768
            self.intermediate_size = 3072


    class DummyViTModel(torch.nn.Module):
        def __init__(self):
            super(DummyViTModel, self).__init__()
            self.config = DummyConfig()
            self.embed_tokens = torch.nn.Embedding(10, 768)
            self.seq_len = 5  # Matching token_mask length


    model = DummyViTModel()

    # Test prune_tokens_vit
    _, pruned_mac, original_mac = prune_tokens_vit(model, token_mask)

    print(f"Original MACs: {original_mac:.4e}")
    print(f"Pruned MACs:   {pruned_mac:.4e}")
    print(f"Reduction:     {100 * (1 - pruned_mac / original_mac):.2f}%")

    # Test mac_per_token
    seq_len_test = 197
    hidden_size_test = 768
    intermediate_size_test = 3072
    num_heads_test = 12

    mac_per_tok = mac_per_token(seq_len_test, hidden_size_test, intermediate_size_test, num_heads_test)
    print(f"MACs saved per token per layer (approx): {mac_per_tok:.4e}")