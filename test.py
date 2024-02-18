from models.mistral import MistralModel

import torch

device = torch.device("mps")
model = MistralModel(device)
print(model.pretty_print())
model.register_hook("layers.31.mlp")
print(
    model.infer(
        messages=[
            {"role": "user", "content": "Lets play two truths and a lie!"},
        ]
    )
)
print(len(model.activations["layers.31.mlp"]))
