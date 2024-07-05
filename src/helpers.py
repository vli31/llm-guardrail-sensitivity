import time
import openai
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

openai.api_key = "FILL-IN-KEY"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(texts, engine="text-embedding-ada-002", model=None, tokenizer=None, batch_size=20):
    """
    Get embeddings for a list of texts using OpenAI's specified engine.

    Args:
    - texts (list of str): The texts to embed.
    - engine (str, optional): The embedding model to use. Default is "text-embedding-ada-002".
    - model (torch.nn huggingface model, optional): The open source embedding model to use. Default is None
    - tokenizer (huggingface tokenizer, optional): The associated tokenizer for the open source embedding model. Default is None
    - batch_size (int, optional): Batch size. Default is 20

    Returns:
    - list of embeddings: A list of embedding vectors for the input texts.
    """
    assert engine or model, "Please specify what embedding model you want to use. For OpenAI model, set `openai` to `True` and provide a valid engine name. \
    For open source model, please pass the model into `embed_model`"
    embeddings = []
    if engine is not None:
        i = 0
        batch = []
        for response in tqdm(texts):
            batch.append(response)
            if i < batch_size:
                i += 1
            else:
                response = openai.Embedding.create(input=batch, engine=engine) # type: ignore
                embeddings.append(response["data"][0]["embedding"])
                i = 0
                batch = []
        if i > 0:
            response = openai.Embedding.create(input=batch, engine=engine) # type: ignore
            embeddings.append(response["data"][0]["embedding"])
    else:
        i = 0
        batch = []
        for response in tqdm(texts):
            batch.append(response)
            if i < batch_size:
                i += 1
            else:
                batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**batch_dict)
                batch_embeddings = outputs.last_hidden_state[:, 0]

                batch_embeddings = F.normalize(embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().detach().numpy()
                embeddings.append(batch_embeddings)
                i = 0
                batch = []
        if i > 0:
            batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(**batch_dict)
            batch_embeddings = outputs.last_hidden_state[:, 0]

            batch_embeddings = F.normalize(embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().detach().numpy()
            embeddings.append(embeddings)
    return embeddings


def generate_persona_response(prompt: str, persona: str|None = None, 
                              temp: float = 1.0, model: str = "gpt-3.5-turbo") -> tuple:
    """
    Generate a response from GPT-3.5 model with a given persona and prompt, 
    with ChatGPT system prompt and appropriate question-answer structure.
    """
    try:
        cur_history = [{"role":"system", "content":"You are ChatGPT, a large language model trained by OpenAI."}]
        if persona is not None: 
            cur_history.append({"role": "user", "content": f"{persona}\n\nCould you help me with a question?"})
            result1 = openai.ChatCompletion.create( # type: ignore
            # result1 = client.chat.completions.create(
                model=model, temperature=temp, messages=cur_history)
            cur_history.append({"role": "assistant", "content": result1.choices[0].message.content})
        
        cur_history.append({"role": "user", "content": f"{prompt}"})
        result2 = openai.ChatCompletion.create( # type: ignore
        # result2 = client.chat.completions.create(
                model=model, temperature=temp, messages=cur_history)
        
        try: 
            if persona is not None: return result1.choices[0].message.content, result2.choices[0].message.content    
            else: return "NO PERSONA", result2.choices[0].message.content    
        
        except Exception as e: 
            return "!!!ERROR!!!", f"{e}" # catch system errors 
        
    except Exception as e:
        print(e)
        time.sleep(25)
        return generate_persona_response(prompt, persona, temp, model)


def generate_response(prompt: str, syst_prompt: str|None = None, temp:float = 1.0, 
                      model: str = "gpt-3.5-turbo", response_format:dict[str, str]={ "type": "text" }):
    """
    Generate a response more generally with a given prompt and system prompt.
    """
    try:
        messages = [] 
        if syst_prompt: messages.append({"role": "system", "content": syst_prompt})
        messages.append({"role": "user", "content": prompt})
        result = openai.ChatCompletion.create( # type: ignore
            model=model, messages=messages, temperature=temp, response_format=response_format)
        
        try: 
            return result.choices[0].message.content
        except Exception as e: 
            return "!!!ERROR!!! " + f"{e}" # catch system errors 
        
    except Exception as e:
        print(e)
        time.sleep(2)
        return generate_response(prompt, syst_prompt, temp,  model)

