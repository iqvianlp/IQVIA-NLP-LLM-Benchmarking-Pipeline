# Open AI models hosted on Azure:
GPT_35_TURBO = "gpt-35-turbo-0613"
GPT_35_TURBO_16k = "gpt-35-turbo-16k-0613"
GPT_4 = "gpt-4"
GPT_4_32k = "gpt-4-32k"

AZURE_DEPLOYMENTS = [
    GPT_35_TURBO,
    GPT_35_TURBO_16k,
    GPT_4,
    GPT_4_32k
]

# Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models @ 29 Sep 2023
MAX_TOKENS = {
    GPT_35_TURBO: 4096,
    GPT_35_TURBO_16k: 16384,
    GPT_4: 8192,
    GPT_4_32k: 32768
}
