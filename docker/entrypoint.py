"""
docker/entrypoint.py
Container bootstrap wrapper for Muse-Glimmer + DFlash Speculative Decoding.
Patches SupportsEagle3 auxiliary layer resolution for multimodal backbones
before delegating to the vLLM OpenAI API server.
"""
import re
import sys

INTERFACES_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/interfaces.py"

try:
    with open(INTERFACES_PATH, "r") as f:
        content = f.read()

    # SupportsEagle3: handle cases where parent_ref is already the inner EagleModelMixin model
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

# Launch the official vLLM OpenAI API Server
from vllm.entrypoints.openai.api_server import main

if __name__ == "__main__":
    main()
