import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'Opisz transformate fourera po polsku'}
    ]
)

print(response['message']['content'])
