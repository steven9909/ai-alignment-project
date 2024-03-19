from abc import ABC, abstractmethod
import typing as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict
import os
from dotenv import load_dotenv


class BaseModel(ABC):
    def __init__(self, name, device, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)

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

    def register_hook(self, layer_name: str, all: bool) -> t.List[str]:
        """Registers a hook to the layer that contains layer_name

        Args:
            layer_name (str): name of the lauer
            all (bool): flag to register to all layers that contain layer_name
        """
        layer_names = []
        for name, module in self.model.named_modules():
            if layer_name in name:
                self.activations[name].clear()
                self.handles.append(
                    module.register_forward_hook(self._get_activation(name))
                )
                layer_names.append(name)
                print(f"Registered hook for {name}")
                if not all:
                    return layer_names

        return layer_names

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


class AuthBaseModel(BaseModel):
    def __init__(self, name, device):
        load_dotenv("./auth.env")
        super().__init__(name, device, token=os.getenv("HUGGINGFACE_TOKEN"))
