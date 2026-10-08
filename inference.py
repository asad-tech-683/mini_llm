from generation.loader import ModelLoader
from generation.generator import Generator
from config import Config


loader = ModelLoader(Config())

model = loader.load_final(
    "D:\\AI-ML\\multi_lang_variants\\32m\\2.0\\final_model.pt"
    # "D:\\AI-ML\\multi_lang_llm\\checkpoints\\checkpoint_step_006000.pt"
    # "checkpoints/final_model.pt"
)

generator = Generator(
    model
)

print(
    generator.generate(
        prompt="انتخابات کے وقت کیے گئے بڑے بڑے وعدوں میں سے کتنے وعدے واقعی پورے ہوتے ہیں، اور کسی حکومت کی کارکردگی جانچنے کا درست طریقہ کیا ہے؟",
        max_new_tokens=100,
        top_k=50,
        temperature=0.8,
    )
)


