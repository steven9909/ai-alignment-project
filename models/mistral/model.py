from ..base import BaseModel
from ..config import ModelConfig
import typing as t


class MistralModel(BaseModel):
    def __init__(self, device):
        super().__init__(ModelConfig.get_config()["mistral"]["name"], device)

    def infer(self, messages: t.List[t.Dict], new_tokens: int = 1):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(
            model_inputs, max_new_tokens=new_tokens, do_sample=True
        )
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0]
