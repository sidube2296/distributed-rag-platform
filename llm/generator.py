import subprocess

def generate_answer(context, question):

    prompt = f"""
You are an intelligent assistant.

Use the context below to answer the question clearly.

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