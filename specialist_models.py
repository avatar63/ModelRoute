import ollama

def load_and_generate(prompt, model_name, keep_alive="5s"):
    resp = ollama.list()
    model_list = [model.model for model in resp.models]

    if model_name not in model_list:
        print(f"Pulling {model_name}...")
        for progress in ollama.pull(model_name, stream=True):
            print(progress.get('status', ''), progress.get('completed', ''), '/', progress.get('total', ''))

    print(f"Using model {model_name}")

    stream = ollama.generate(
        model=model_name,
        prompt=prompt,
        keep_alive=keep_alive,
        stream=True,
    )

    for chunk in stream:
        yield chunk["response"]