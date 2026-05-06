import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
print('Key prefix:', os.environ['GROQ_API_KEY'][:6] + '…')
client = Groq(api_key=os.environ['GROQ_API_KEY'])
resp = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{'role':'user','content':'Say hi.'}],
)
print('Response:', resp.choices[0].message.content)