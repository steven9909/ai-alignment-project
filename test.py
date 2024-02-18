from models.mistral.model import MistralModel

import torch

device = torch.device("mps")
model = MistralModel(device)
print(
    model.infer(
        messages=[
            {"role": "user", "content": "What is your favourite condiment?"},
            {
                "role": "assistant",
                "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
            },
            {"role": "user", "content": "Do you have mayonnaise recipes?"},
        ]
    )
)
