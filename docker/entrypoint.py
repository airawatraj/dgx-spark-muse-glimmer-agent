"""
docker/entrypoint.py
Container bootstrap wrapper for Muse-Glimmer + DFlash Speculative Decoding.
Patches:
1. SupportsEagle3 auxiliary layer resolution for multimodal backbones
2. DFlash SWA sliding_window fallback
3. DFlash encoder.* weight key mapping to model.fc / model.hidden_norm
before delegating to the vLLM CLI.
"""
import re
import sys

INTERFACES_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/interfaces.py"
QWEN3_DFLASH_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py"

# 1. Patch interfaces.py for SupportsEagle3 multimodal backbones
try:
    with open(INTERFACES_PATH, "r") as f:
        content = f.read()

    pattern = r'assert hasattr\(parent_ref, "model"\).*?parent_ref\.model\._set_aux_hidden_state_layers\(layers\)'
    replacement = """inner_model = getattr(parent_ref, "model", parent_ref)
        assert isinstance(inner_model, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        inner_model._set_aux_hidden_state_layers(layers)"""

    patched = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if patched != content:
        with open(INTERFACES_PATH, "w") as f:
            f.write(patched)
except Exception as e:
    print(f"[entrypoint.py] Warning: Could not patch interfaces.py: {e}", file=sys.stderr)

# 2. Patch qwen3_dflash.py for sliding_window fallback and weight name mapping
try:
    with open(QWEN3_DFLASH_PATH, "r") as f:
        content = f.read()

    target_err = 'raise ValueError(\n                "DFlash sliding attention requires a window size configured in "\n                "dflash_config.swa_window_size or the top-level sliding_window."\n            )'
    if target_err in content:
        content = content.replace(target_err, "sliding_window = 4096")

    target_loop = 'if "t2d" in name:\n                continue'
    replacement_loop = """if "t2d" in name:
                continue
            if name.startswith("encoder.fc"):
                name = name.replace("encoder.fc", "fc")
            elif name.startswith("encoder.output_norm_enc"):
                name = name.replace("encoder.output_norm_enc", "hidden_norm")"""
    if target_loop in content and "encoder.fc" not in content:
        content = content.replace(target_loop, replacement_loop)

    with open(QWEN3_DFLASH_PATH, "w") as f:
        f.write(content)
except Exception as e:
    print(f"[entrypoint.py] Warning: Could not patch qwen3_dflash.py: {e}", file=sys.stderr)

# Ensure sys.argv starts with ['vllm', 'serve']
if len(sys.argv) > 1 and sys.argv[1] != "serve":
    sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
