import backoff
import openai
import json

######################################################### API Setup #########################################################
# Initialize OpenAI client
client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

# Low temp short response, the initial query
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def api_call_initial(
        msg,
        model,
        system_message,
        temperature=0.5
):
    print("Sending API call:", "system message:", system_message, "user message:", msg)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
    )
    print("Recieved response:", response.choices[0].message.content)
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict

# High temp long response, the followup query
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def api_call_followup(
        chat_log,
        model,
        temperature=0.8
):
         
    print("Sending API call:", chat_log)
    response = client.chat.completions.create(
        model=model,
        messages=chat_log,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    print("Recieved response:", response.choices[0].message.content)
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict