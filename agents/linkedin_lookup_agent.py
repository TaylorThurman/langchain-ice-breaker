from tools.tools import get_profile_url

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given the full name {name_of_person} I want you to get me a link to their LinkedIn profile page. 
    Your answer should contain only a URL."""

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the LinkedIn profile page of a person",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
