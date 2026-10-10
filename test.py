

import ollama

GENERAL_SYSTEM_PROMPT = (
    "You are the general-purpose assistant inside a specialist-routing AI harness. "
    "Specific domains (code, legal questions, and others as they're added) are handled "
    "by dedicated fine-tuned specialist models, not you — you are only invoked when a "
    "query doesn't match any of them. Answer helpfully and conversationally within your "
    "own general knowledge. If a question is clearly better suited to a domain specialist "
    "that may exist, you can mention that, but you should still do your best to answer directly."
    # f"You have the following experts at your desposal: {REGISTRY}"
)


resp = ollama.chat(
    model="qwen3:4b",
    messages=[
        {"role": "system", "content": "You must respond only in French, regardless of what language the user writes in. This is absolute."},
        {"role": "user", "content": "hello, how are you"},
    ],
)
# print(resp["message"]["content"])
print(resp["message"]["content"])