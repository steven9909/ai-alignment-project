from omegaconf import OmegaConf


class ModelConfig:
    conf = OmegaConf.load("./models/config.yaml")

    @classmethod
    def get_config(cls):
        return cls.conf
