import birder

models = birder.list_pretrained_models()

print(f"Found {len(models)} pretrained models:\n")

for m in models:
    print(m)
