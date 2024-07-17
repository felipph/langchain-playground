from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Executa quando o LLM começa a rodar"""
        print(f"*** Prompt para o LLM foi: ***\n{prompts[0]}")
        print("******")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Roda quando o LLM termina de rodar"""
        print(f"*** Resposta da LLM foi: ***\n{response.generations[0][0].text}")
        print("******")