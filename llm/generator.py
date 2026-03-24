import subprocess

def generate_answer(context, query):

    prompt = f"""
You are an AI assistant.

Answer the question ONLY based on the context below.
If the context is not relevant, say "I don't know".

Context:
{context}

Question:
{query}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()