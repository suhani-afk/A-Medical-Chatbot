from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use the retrieved context to answer the question. "
     "If you don't know the answer, say you don't know.\n\n"
     "{context}"
    ),
    ("human", "{input}")
])
