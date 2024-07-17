from typing import Union, List

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import tool

from callback import AgentCallbackHandler
import re

load_dotenv();


@tool
def get_text_length(text: str) -> int:
    """Returns the length of provided text """
    # return len(text)
    pattern = re.compile('[\W_]+')
    normalized_text = pattern.sub('', text)
    return len(normalized_text)


def find_tool_by_name(tools: List, tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Can't find tool with name {tool_name}!")


if __name__ == "__main__":
    print("ReAct Agent")

    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools),
                                                                     tool_names=", ".join(t.name for t in tools))

    llm = ChatGroq(
        temperature=0,
        # model="mixtral-8x7b-32768",
        model="llama3-70b-8192",
        # model="gemma2-9b-it",
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()]
    )

    intermediate_steps = []

    agent = ({
                 "input": lambda x: x["input"],
                 "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
             }
             | prompt
             | llm
             | ReActSingleInputOutputParser()
             )
    agent_step = None
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of word: Calado?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(tool_input)
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
    # print(agent_step)
#
# chat = ChatGroq(
#     temperature=0,
#     model="llama3-70b-8192",
# )
#
# system = "You are a helpful assistant."
# human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
#
# chain = prompt | chat
# response = chain.invoke({"text": "Explain the importance of low latency for LLMs."})
# print(response)
