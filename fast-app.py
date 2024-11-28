from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Date configuration
now = datetime.now()
current_date = now.date()

# LangChain models and retriever setup
gpt_mini = ChatOpenAI(model="gpt-4o-mini")
retriever = PineconeVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
).as_retriever(
    k=10,
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.1},
)

# Prompt templates
training_template = """Answer based on the given context and question.

INSTRUCTIONS:
- Answer questions based solely on the provided meeting transcript context
- If the answer is not found in the context, state that the information is not available
- Today's date is {current_date} - use for time-related queries

Question: {question}
Context: {context}

Answer:"""

rephrase_template = """
You are given a conversation history and a follow-up question. Your task is to rephrase the follow-up question into a standalone question.

**Chat History:**
{chat_history}

**Follow-Up Question:**
{question}

**Rephrased Standalone Question:**
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(rephrase_template)
rephrase_chain = CONDENSE_QUESTION_PROMPT | gpt_mini | StrOutputParser()

# Request and Response Models
class OnMessageRequest(BaseModel):
    message: str
    chat_history: List[str] = []

class OnMessageResponse(BaseModel):
    response: str


@app.post("/on_message", response_model=OnMessageResponse)
async def on_message(request: OnMessageRequest):
    try:
        # Rephrase the question
        rephrased_question = rephrase_chain.invoke(
            {
                "question": request.message,
                "current_date": current_date,
                "chat_history": request.chat_history,
            }
        )

        # Retrieve relevant context
        context = retriever.invoke(rephrased_question)
        if not context:
            raise ValueError("No relevant context found.")

        # Generate final answer
        prompt = ChatPromptTemplate.from_template(training_template)
        chain = prompt | gpt_mini | StrOutputParser()

        answer = chain.invoke(
            {
                "question": rephrased_question,
                "current_date": current_date,
                "context": context,
            }
        )

        # Update chat history
        request.chat_history.append(HumanMessage(content=request.message))
        request.chat_history.append(AIMessage(content=answer))

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the message: {str(e)}")
