# LLM Clients
Package with clients to connect to hosted LLMs:
- OpenAI models hosted on Azure
- Hugging Face models served with TGI (Text Generation Inference) or vLLM

## Creating servers
The following links have information on spinning up servers for different clients:
- [TGIClient](https://github.com/huggingface/text-generation-inference)
- [VLLMClient](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)

###Usage
1. If using Hugging Face-based models, start the server.
2. Instantiate client:
    ```python
    from tools.llm.clients import AzureClient, TGIClient, VLLMClient
    
    llm = AzureClient(model_name)
    
    # For Hugging Face based client, ensure chat_completion_enabled is set to True if the model is a chat/instruction model,
    # otherwise the chat_template won't be used.
    chat_completion_enabled = True # if chat model, else False
    
    llm = TGIClient(ip_or_url, model_name, chat_completion_enabled=chat_completion_enabled)
    llm = VLLMClient(ip_or_url, model_name, chat_completion_enabled=chat_completion_enabled)
    ```
3. Check LLM is up and running:
    ```python
    llm.is_alive()
    ```
4. Check input fits the model:
    ```python
    llm.is_valid_prompt(prompt)
    ```
5. Prompt LLM:
   - If the model is a chat model (i.e. it is an OpenAI chat model or `chat_completion_enabled` is `True`), the prompt
     may be:
     - A string (which will be automatically converted to a user message):
         ```python
         llm('Hey there!')
         ```
     - A list of message dictionaries with *role* and *content* keys:
         ```python
         messages = [
             {'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user', 'content': 'Hey there!'},
             {'role': 'assistant', 'content': 'Hi! How can I help you?'},
             {'role': 'user', 'content': 'What is 2+2?'},
         ]
         llm(messages)
         ```
       Note that all models support all roles.
   - If the model is not chat-enabled, only strings are accepted as prompts.
