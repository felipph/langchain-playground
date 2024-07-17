from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_groq import ChatGroq

from main import get_text_length
load_dotenv()

if __name__ == "__main__":
    llm = ChatGroq(
        # model="mixtral-8x7b-32768", ## esse foi mais duro com a resposta da função
        # model="llama3-70b-8192", ## esse inventou uma resposta
        # model="llama3-8b-8192", ## esse entrou em loop
        model="gemma2-9b-it", ## Esse limpou a entrada antes de passar para a função e ficou correto
        temperature=0
    )

    agent_executor: AgentExecutor = initialize_agent(
        tools=[get_text_length],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    agent_executor.invoke({"input": "Qual o tamanho da palavra 'Borges'?"})


