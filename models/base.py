from abc import ABC, abstractmethod
import typing as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict


class BaseModel(ABC):
    def __init__(self, name, device):
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self.handles = []
        self.activations = defaultdict(list)
        self.device = device

    @abstractmethod
    def infer(self, *_: t.Any) -> t.Any:
        """Run inference on model

        Returns:
            t.Any: any parameters required for inference
        """
        pass

    def pretty_print(self):
        """Pretty prints module names"""
        for name, _ in self.model.named_modules():
            print(name)

    def register_hook(self, layer_name: str) -> None:
        """Registers a hook to the layer that contains layer_name

        Args:
            layer_name (str): name of the lauer
        """
        for name, module in self.model.named_modules():
            if layer_name in name:
                self.activations[name].clear()
                self.handles.append(
                    module.register_forward_hook(self._get_activation(layer_name))
                )
                print(f"Registered hook for {name}")
                break

    def clear_all_hooks(self) -> None:
        """Clears all hooks that have been registered on this model"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("Cleared all hooks")

    def _get_activation(
        self, name: str
    ) -> t.Callable[[t.Any, t.Any, torch.Tensor], None]:
        """Returns a hook that stores the activation of the layer in self.activations

        Args:
            name (str): name of the layer

        Returns:
            t.Callable[[t.Any, t.Any, torch.Tensor], None]: hook function
        """

        def hook(_1, _2, output):
            self.activations[name].append(output.detach().cpu().numpy())

        return hook
