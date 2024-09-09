import abc
import logging
from datetime import datetime, timedelta
from typing import Any, Sequence, Union

import requests
from azure.identity import ClientSecretCredential
from langchain.schema import HumanMessage
from langchain_community.adapters.openai import convert_openai_messages
from langchain_openai import AzureChatOpenAI
from openai import OpenAI, APIError, APIConnectionError
from transformers import AutoTokenizer

from tools.llm.constants import *
from tools.llm.deployments import AZURE_DEPLOYMENTS, MAX_TOKENS


LOGGER = logging.getLogger(__name__)

PromptType = Union[str, Sequence[dict[str, str]]]


class BaseLLMClient(metaclass=abc.ABCMeta):
    """
    Abstract base class for prompting LLMs.

    Attributes:
        model_name (str): Name of the model used by this client.
        chat_completion_enabled (bool): Whether chat completion is enabled for the model(s) this client is prompting.
        _timeout (int): Timeout to use when prompting.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.chat_completion_enabled = False
        self._timeout = 30

    def __repr__(self):
        return f'model={self.model_name}; max_tokens={self.model_max_length}; url={self._url}'

    @abc.abstractmethod
    def __call__(self, prompt: PromptType, **kwargs: dict[str, Any]) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_max_length(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _url(self):
        raise NotImplementedError

    def is_valid_prompt(self, prompt: str):
        tokens = self._token_count(prompt)
        return tokens, tokens <= self.model_max_length

    @abc.abstractmethod
    def _token_count(self, prompt: PromptType):
        raise NotImplementedError

    @abc.abstractmethod
    def is_alive(self):
        raise NotImplementedError


class BaseHuggingFaceLLMClient(BaseLLMClient):
    """
    Abstract base class implementing some common methods used by clients that interact with Hugging Face models.
    """

    def __init__(self, model_name, chat_completion_enabled: bool = False):
        """
        Initializes a new BaseHuggingFaceLLMClient object.

        Args:
            model_name: Hugging Face model ID.
            chat_completion_enabled: Whether to treat the model as a chat model.
        """
        super().__init__(model_name)
        self.chat_completion_enabled = chat_completion_enabled
        self._tokenizer = self._determine_tokenizer()

    def _determine_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except RecursionError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                'Error loading tokenizer; can the host access HuggingFace?')
        return tokenizer

    def _token_count(self, prompt: PromptType) -> int:
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return len(self._tokenizer(prompt)['input_ids'])


class TGIClient(BaseHuggingFaceLLMClient):
    """
    Client for communicating with a Text Generation Inference server.

    Attributes:
        _base_url (str): Base URL of the TGI server.
        _model_max_length (int): Max length the model can support.
    """

    def __init__(self, ip: str, model_name: str, port: int = 8080, chat_completion_enabled: bool = False):
        """
        Initializes a new TGIClient object.

        Args:
            ip: IP address of the text-generation-inference server.
            model_name: Name of the model deployed on the server.
            port: Port of the server.
            chat_completion_enabled: Whether to treat the model as a chat model.
        """
        super().__init__(model_name, chat_completion_enabled)
        if ip.startswith('http'):
            self._base_url = f'{ip.rstrip("/")}:{port}'
        else:
            self._base_url = f'http://{ip.rstrip("/")}:{port}'
        self._verify_model_id(model_name)
        self._model_max_length = self._get_info().json()['max_input_length']

    def __call__(self, prompt: PromptType, params: dict = None) -> str:
        """
        Generates text using the given prompt and generation params.

        Args:
            prompt: Text to submit to the LLM.
            params: Generation params to submit to the LLM.

        Returns:
            Generated text.

        Raises:
            LLMClientException: Text generation failed due to a server error.
        """
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        response = self._post(prompt, params)
        try:
            if response.status_code == 200:
                return response.json()['generated_text']
            else:
                raise LLMClientException(
                    f'Generation failed with code {response.status_code}: {response.json()["error"]}'
                )
        except requests.exceptions.JSONDecodeError:
            raise LLMClientException(f'Generation failed with error: {response.text}')

    def _post(self, prompt: str, params: dict = None) -> requests.Response:
        params = params if params else {'max_new_tokens': 100}
        return requests.post(
            self._url,
            json={'inputs': prompt, 'parameters': params},
            timeout=self._timeout
        )

    @property
    def model_max_length(self):
        return self._model_max_length

    @property
    def _url(self):
        return f'{self._base_url}/generate'

    @property
    def _info_url(self):
        return f'{self._base_url}/info'

    def _get_info(self) -> requests.Response:
        return requests.get(self._info_url)

    def _verify_model_id(self, model_name) -> None:
        """
        Sanity check to ensure the deployed model is the same as the expected model.

        Returns:
            None

        Raises:
            LLMClientException: The server can't be reached or the served LLM is different from what was used to
                initialize this object.
        """
        response = self._get_info()
        if response.status_code == 200:
            if model_name != response.json()['model_id']:
                raise LLMClientException(
                    'The model being served does not match the model used to initialize this object.'
                )
        else:
            raise LLMClientException('The endpoint cannot be reached.')

    def is_alive(self) -> bool:
        try:
            response = self._get_info()
            return response.status_code == 200
        except requests.exceptions.ConnectTimeout:
            return False


class VLLMClient(BaseHuggingFaceLLMClient):
    """
    Client for communicating with a vLLM server.

    Attributes:
        _base_url (str): Base URL of the vLLM server.
        _model_max_length (int): Max length the model can support.
    """

    def __init__(self, url: str, model_name: str, port: int = 443, chat_completion_enabled: bool = False,
                 api_key: str = 'none'):
        """
        Initializes a new VLLMClient object.

        Args:
            url: URL of the vLLM server.
            model_name: Name of the model deployed on the server.
            port: Port of the server.
            chat_completion_enabled: Whether to treat the model as a chat model.
        """
        super().__init__(model_name, chat_completion_enabled)
        if url.startswith('http'):
            self._base_url = f'{url.rstrip("/")}:{port}'
        else:
            self._base_url = f'http://{url.rstrip("/")}:{port}'
        self._connector = OpenAI(
            base_url=self._url,
            api_key=api_key
        )
        self._verify_model_id(model_name)
        self._model_max_length = self._get_model_list()[0].max_model_len

    def __call__(self, prompt: PromptType, params: dict = None) -> str:
        """
        Generates text using the given prompt and generation params.

        Args:
            prompt: Text to submit to the LLM.
            params: Generation params to submit to the LLM.

        Returns:
            Generated text.

        Raises:
            LLMClientException: Text generation failed due to a server error.
        """
        kwargs = {'max_tokens': 100, 'temperature': 0}

        if params:
            kwargs.update(params)

        try:
            if self.chat_completion_enabled:
                if isinstance(prompt, str):
                    messages = [{'role': 'user', 'content': prompt}]
                elif len(prompt) > 0 and isinstance(prompt[0], dict):
                    messages = prompt
                else:
                    messages = []
                response = self._connector.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                generated_text = response.choices[0].message.content
            else:
                response = self._connector.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
                generated_text = response.choices[0].text
            return generated_text
        except APIError as e:
            raise LLMClientException(e.message)

    @property
    def model_max_length(self):
        return self._model_max_length

    @property
    def _url(self):
        return f'{self._base_url}/v1'

    @property
    def _model_url(self):
        return f'{self._url}/models'

    def _get_model_list(self) -> list:
        """
        Returns the list of models available of the server.
        """
        return self._connector.models.list().data

    def _verify_model_id(self, model_name) -> None:
        """
        Sanity check to ensure the deployed model is the same as the expected model.

        Returns:
            None

        Raises:
            LLMClientException: The server can't be reached or the served LLM is different from what was used to
                initialize this object.
        """
        try:
            model_list = self._get_model_list()
            if model_name != model_list[0].id:
                raise LLMClientException(
                    'The model being served does not match the model used to initialize this object.'
                )
        except APIConnectionError:
            raise LLMClientException('The endpoint cannot be reached.')

    def is_alive(self) -> bool:
        try:
            self._get_model_list()
            return True
        except LLMClientException:
            return False


class AzureClient(BaseLLMClient):

    def __init__(self, model_name, chat_completion_enabled: bool = False):
        if model_name not in AZURE_DEPLOYMENTS:
            raise ValueError(f'Model "{model_name}" is invalid; please choose one of:\n{", ".join(AZURE_DEPLOYMENTS)}')
        super().__init__(model_name)
        self._token_requester = ClientSecretCredential(
                TENANT_ID,
                SERVICE_PRINCIPAL,
                SERVICE_PRINCIPAL_SECRET
            )
        self._connect()
        self.chat_completion_enabled = chat_completion_enabled

    def _connect(self):
        self._token_requested_at = datetime.now().replace(microsecond=0)
        self._token_object = self._token_requester.get_token(SCOPE_NON_INTERACTIVE)
        LOGGER.info(f'New token will expire at {self._token_requested_at + timedelta(hours=1)} (client time; '
                    f'server time: {datetime.fromtimestamp(self._token_object.expires_on + 3600)})')
        self._connector = AzureChatOpenAI(
            azure_deployment=self.model_name,
            azure_endpoint=self._url,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=self._token_object.token,
            openai_api_type='azure_ad'
        )

    def __call__(self, prompt: PromptType, **kwargs: dict[str, Any]) -> str:
        """
        Generates text using the given prompt.

        Args:
            prompt: Text to submit to the LLM or a list of dicts with "role" and "content" keys.
            kwargs: Unused.

        Returns:
            Generated text.
        """
        if self._token_expired():
            LOGGER.info('Azure token expired; requesting new one')
            self._connect()
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        elif len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = convert_openai_messages(prompt)
        else:
            messages = []
        return self._connector.invoke(messages).content

    def _token_expired(self):
        """Start requesting a new token just before the 1h expiry window."""
        return datetime.now() >= self._token_requested_at + timedelta(minutes=59)

    @property
    def model_max_length(self):
        return MAX_TOKENS[self.model_name]

    @property
    def _url(self):
        return OPENAI_API_BASE

    def _token_count(self, prompt):
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            return self._connector.get_num_tokens_from_messages(convert_openai_messages(prompt))
        else:
            return self._connector.get_num_tokens(prompt)

    def is_alive(self):
        return self._token_object.token is not None


class LLMClientException(Exception):
    pass


def max_length_from_configs(configs):
    model_max_length = 512
    for mml_key in ["n_positions", "max_sequence_length", "max_position_embeddings"]:
        if hasattr(configs, mml_key):
            model_max_length = getattr(configs, mml_key)
    return model_max_length
