import ollama





def load_and_generate(prompt, model_name):

    resp = ollama.list()
    model_list = [model.model for model in resp.models]
    print(f"Using model {model_name}")    
    if model_name not in model_list:
        for progress in ollama.pull(model_name, stream=True):
            print(progress.get('status', ''), progress.get('completed', ''), '/', progress.get('total', ''))

    response = ollama.generate(
        model=model_name,
        prompt=prompt
    )
    return response["response"]




print(load_and_generate("I want all the data in my employees table.", "hf.co/MaziyarPanahi/sqlcoder-7b-2-GGUF:Q4_K_M"))
