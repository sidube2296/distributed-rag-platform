import subprocess

def generate_answer(context, question):

    prompt = f"""
Context:
{context}

Question:
{question}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout