from dotenv import load_dotenv
import chainlit as cl

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.globals import set_debug, set_verbose
from datetime import datetime

now = datetime.now()
current_date = now.date()

set_debug(True)
set_verbose(True)

load_dotenv()

policy_template = """Please analyze the following question and context to provide a response based on the given information.

Role: As a Cogent Labs assistant bot, your primary role is to provide information about Cogent Labs' policies and training guidance. 

ANSWERING INSTRUCTIONS:
- For greetings such as "Hi," "Hello," or similar, respond in a polite and friendly manner, acknowledging the user and offering assistance.
- For farewells such as "Goodbye,","Bye", "See you later," or similar, respond courteously, wishing the user well and offering further help if needed.
- For other general conversational queries or small talk, respond appropriately to maintain engagement but stay focused on providing relevant information.
- Ensure that all policies pertain specifically to Cogent Labs. If the question involves a different company, do not provide an answer.
- Respond gently for general inquiries, maintaining an appropriate assistant-like tone, but remain focused on the context.
- Include link names as the document page names in markdown format.
- Address the question directly based on the policy, without including document names.
- Always provide a link to the relevant policy for further information.
- Provide email addresses or contact information only if they are mentioned in the context; otherwise, omit them.
- Avoid using phrases like "Based on the provided context." Simply provide the answer.
- If the context does not contain sufficient information to answer the question, state that the information is not present in Cogent Labs' policies. Provide this link for user to get more detail Cogent-wiki-Link=http://wiki.cogentlabs.co/
- Today's date is {current_date}. Please use this date as a reference when answering any time-related queries. For example, if asked about the most recent company updates, ensure that your response considers updates up to and including today’s date.te today so make sure to refer to it when you are answering time realted queries like  what are most recent compnay updates

Question: {question}
Context: {context}

Answer:"""

training_template = """Analyze the given question and context and answer the question accordingly.

INSTRUCTIONS FOR ANSWER:
- For greetings such as "Hi," "Hello," or similar, respond in a polite and friendly manner, acknowledging the user and offering assistance.
- For farewells such as "Goodbye,","Bye", "See you later," or similar, respond courteously, wishing the user well and offering further help if needed.
- For other general conversational queries or small talk, respond appropriately to maintain engagement but stay focused on providing relevant information.
- Ensure that all queries pertain specifically to Cogent Labs. If the question involves a different company, do not provide an answer.
- Respond gently for general inquiries, maintaining an appropriate assistant-like tone, but remain focused on the context.
- Describe the training progarm
- Provide Neccessary Resource Links if provided in the context .
- If you DONT know the answer within context, don't answer the question
- please refrain from adding document name as reference in the answer.
- Only Add resources that are relevant and whoe links are provided otherwise don't add resources. also make sure the link works if they don't work don't add them.
- Always Provide Link to the relevant training roadmap so that user can get more information regarding it
- refrain from adding this line "Based on the provided context." just give the answer.
- If the context does not contain sufficient information to answer the question, state that the information is not present in Cogent Labs' policies. Provide this link for user to get more detail Cogent-wiki-Link=http://wiki.cogentlabs.co/
- Today's date is {current_date}. Please use this date as a reference when answering any time-related queries. For example, if asked about the most recent company updates, ensure that your response considers updates up to and including today’s date.te today so make sure to refer to it when you are answering time realted queries like  what are most recent compnay updates

Question: {question}
Context: {context}

Answer:"""


routing_template = """
Given the question, determine whether the question is related to a policy or an engineer training program.
Make sure you extract essence of the question, analyze the question from all possible ways before classifying it. Take time to understand the nuances of question by evaluating it yourself. Here are the following scenarios:
- Case 1: IF the question is related to training and guidance for engineering roadmaps then return "Training"
- Case 2: IF the question is about policy and any other information return "Policy"

Question: {question}

Use this format instruction for its output:
{format_instructions}

Response:"""


gpt_mini = ChatOpenAI(model="gpt-4o-mini")


rephrase_template = """
You are given a conversation history and a follow-up question. Your task is to rephrase the follow-up question into a standalone question. Consider the following guidelines when rephrasing:

1. **Date Relevance:** Today's date is {current_date}. If the follow-up question involves any time-related aspects, ensure that your rephrasing considers events, updates, or information up to and including today’s date.

2. **Greetings and Farewells:**
   - For greetings such as "Hi," "Hello," or similar, respond politely and friendly, acknowledging the user and offering assistance.
   - For farewells such as "Goodbye," "Bye," "See you later," or similar, respond courteously, wishing the user well and offering further help if needed.

3. **General Conversational Queries:** For other general conversational queries or small talk, provide a response that maintains engagement while staying focused on delivering relevant information.

**Chat History:**
{chat_history}

**Follow-Up Question:**
{question}

**Rephrased Standalone Question:**
"""


retriever = PineconeVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
).as_retriever(
    k=10,
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.1},
)


class router(BaseModel):
    choice: str = Field(
        descripion="""Make sure you extract essence of the question, analyze the question from all possible ways before classifying it. Take time to understand the nuances of question by evaluating it yourself. Here are the following scenarios:
                    - Case 1: IF the question is related to training and guidance for engineering roadmaps then return "Training"
                    - Case 2: IF the question is about policy and any other information return "Policy" """
    )


parser = PydanticOutputParser(pydantic_object=router)
ROUTING_PROMPT = PromptTemplate(
    template=routing_template,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(rephrase_template)
routing_prompt = PromptTemplate.from_template(routing_template)
rephrase_chain = CONDENSE_QUESTION_PROMPT | gpt_mini | StrOutputParser()
routing_chain = ROUTING_PROMPT | gpt_mini | parser


async def rephrase_chain_func(question, history):
    return rephrase_chain.invoke(
        {"question": question, "current_date": current_date, "chat_history": history}
    )


async def routing_chain_func(question):
    return routing_chain.invoke({"question": question})


async def retriever_func(msg, question):
    if not isinstance(question, str) or question.strip() == "":
        raise ValueError("Invalid question provided. It must be a non-empty string.")
    try:
        result = retriever.invoke(question)
    except Exception as e:

        await msg.send()
        msg.content = (
            "Error while retrieving data server overloaded please try again after 10s"
        )
        await msg.update()
        raise RuntimeError(f"An error occurred while invoking the retriever: {e}")

    return result


async def retrieval_chain_func(route):
    if route.choice == "Policy":
        template = policy_template
    elif route.choice == "Training":
        template = training_template
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | gpt_mini | StrOutputParser()
    return chain


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Medical Reimbursement Eligibility",
            message="Are team members on probation or notice period eligible for medical bill reimbursement?",
            icon="/public/policy.svg",
        ),
        cl.Starter(
            label="Consecutive Casual Leaves",
            message="Can casual leaves be taken consecutively for more than 2 days?",
            icon="/public/policy2.svg",
        ),
        cl.Starter(
            label="Backend Engineer Training Roadmap",
            message="What is the training roadmap for a backend engineer?",
            icon="/public/road_map.svg",
        ),
        cl.Starter(
            label="Notice Period Start Date",
            message="When does the notice period officially begin after submitting a resignation?",
            icon="/public/policy2.svg",
        ),
    ]


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    try:
        msg = cl.Message(content="")
        await msg.send()

        question = await rephrase_chain_func(message.content, chat_history)
        route = await routing_chain_func(question)
        context = await retriever_func(msg, question)
        chain = await retrieval_chain_func(route)
        if isinstance(chain, str):
            for part in chain:
                await msg.stream_token(part)
        else:
            async for part in chain.astream(
                {
                    "question": question,
                    "current_date": current_date,
                    "context": context,
                    "chat_history": chat_history,
                }
            ):
                await msg.stream_token(part)
        chat_history.extend(
            [HumanMessage(content=message.content), AIMessage(content=msg.content)]
        )
        await msg.update()
    except:
        await msg.send()
        msg.content = "Sorry Bot is overloaded try again later for queries"
        await msg.update()
