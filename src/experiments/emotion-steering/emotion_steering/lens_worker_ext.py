"""Named worker RPC for selected-layer hooks; no callable serialization needed."""
from vllm_lens._worker_ext import HiddenStatesExtension

def install_selected_hooks(worker, selected_layers):
    """Keep Lens post-hooks only at frozen intervention layers on every TP rank."""
    from vllm_lens._worker_ext import _get_layers
    layers = _get_layers(worker.model_runner.model)
    before = [(set(layer._forward_hooks), set(layer._forward_pre_hooks)) for layer in layers]
    worker.install_hooks()
    selected = set(selected_layers)
    for index, (layer, (post, pre)) in enumerate(zip(layers, before)):
        # This backend never uses pre-hooks or activation capture.
        for key in set(layer._forward_pre_hooks) - pre:
            del layer._forward_pre_hooks[key]
        if index not in selected:
            for key in set(layer._forward_hooks) - post:
                del layer._forward_hooks[key]
    worker._should_capture = False
    return sorted(selected)


class SelectedLayerExtension(HiddenStatesExtension):
    def install_selected_hooks(self, selected_layers):
        return install_selected_hooks(self, selected_layers)
