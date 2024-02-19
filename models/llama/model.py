from ..base import AuthBaseModel
from ..config import ModelConfig
import typing as t


class LlamaModel(AuthBaseModel):
    def __init__(self, device):
        super().__init__(ModelConfig.get_config()["llama"]["name"], device)

    def infer(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(inputs.input_ids, max_length=1000)

        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
