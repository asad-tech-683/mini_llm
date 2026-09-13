from generation.loader import ModelLoader
from generation.generator import Generator
from config import Config


loader = ModelLoader(Config())

model = loader.load_checkpoint(
    "checkpoints/checkpoint_step_000010.pt"
)

generator = Generator(
    model
)

print(
    generator.generate(
        prompt="پاکستان ہے",
        max_new_tokens=100,
        top_k=50,
        temperature=0.8,
    )
)