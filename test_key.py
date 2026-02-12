import os
from openai import OpenAI

# Paste your key here for the test
os.environ["OPENAI_API_KEY"] = "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" 

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say 'System Ready' if you hear me."}]
    )
    print("✅ SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("❌ ERROR:", e)