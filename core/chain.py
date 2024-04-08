from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains import LLMChain


class Chain(LLMChain):
    """Class responsible for creating LLM chain with llm and prompt as attributes"""

    def __init__(
            self,
            llm: BaseChatModel,
            prompt: BasePromptTemplate
    ):
        super().__init__(llm=llm, prompt=prompt)
