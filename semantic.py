
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from typing import Literal

# Load environment variables from .env file (need OPENAI_API_KEY)
_ = load_dotenv()

# Basic user facts - this could come from a database in a real system if user authentication is implemented
profile = {
    "name": "John",
    "full_name": "John Doe", 
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# Triage rules and agent instructions - these define how emails are classified
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates", 
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

# Initialize the LLM - this connects to OpenAI's API using the key from .env file
llm = init_chat_model("openai:gpt-4o-mini")


# Router class for structured output
# This inherits from BaseModel (Pydantic) to ensure the LLM returns data in exactly the format we expect
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""
    
    # reasoning uses Pydantic's Field to provide a description of the reasoning behind the classification.
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    # classification is a field that uses Literal to restrict its values to specific strings.
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

# Now LLM is set up to return structured output conforming to the Router model.
# Behind the scenes, LangChain adds JSON schema instructions to the system prompt
llm_router = llm.with_structured_output(Router)

# Define the triage system prompt template (we will fill in the profile and triage rules dynamically)
triage_system_prompt = """You are an intelligent email assistant for {full_name} ({name}), who is a {user_profile_background}.

Your task is to analyze incoming emails and classify them into one of three categories:

CLASSIFICATION RULES:
- IGNORE: {triage_no}
- NOTIFY: {triage_notify}  
- RESPOND: {triage_email}

USER CONTEXT:
- Name: {name}
- Background: {user_profile_background}
- You should consider the sender, subject, and content when making decisions
- Always provide clear reasoning for your classification

Be thorough in your reasoning and choose the most appropriate classification."""

# Define the user prompt template
triage_user_prompt = """Please analyze this email and classify it:

FROM: {author}
TO: {to}
SUBJECT: {subject}

EMAIL CONTENT:
{email_thread}

Provide your classification with detailed reasoning."""


# The @tool decorator tells LangChain that this function can be called by the LLM.
# We mock the return values to simulate real functionality.

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email via SMTP or email service API
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule via Google Calendar API, Outlook, etc.
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar via API
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


# Import the memory storage system from LangGraph
from langgraph.store.memory import InMemoryStore

# Create an in-memory store for semantic memory (local vector database)
# The "embed" index means it will use embeddings to find similar information (like RAG)
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}  # Uses OpenAI's embedding model to create vectors
)

# Import memory management tools from LangMem (LangChain's memory library)
from langmem import create_manage_memory_tool, create_search_memory_tool

# Create a tool that lets the agent STORE information in memory
# namespace is like organizing memory into folders - this creates a unique space for each user
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant",           # App name
        "{langgraph_user_id}",      # User ID (will be filled in later)
        "collection"                # Type of memory (semantic facts)
    )
)

# Create a tool that lets the agent SEARCH for information in memory
# Uses the same namespace so it searches in the right user's memory
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",           # App name  
        "{langgraph_user_id}",      # User ID (will be filled in later)
        "collection"                # Type of memory (semantic facts)
    )
)

# Let's examine what these memory tools can do:
print("=== MEMORY TOOL INFORMATION ===")
print(f"Manage tool name: {manage_memory_tool.name}")
print(f"Manage tool description: {manage_memory_tool.description}")
print(f"Manage tool arguments: {manage_memory_tool.args}")
print()
print(f"Search tool name: {search_memory_tool.name}")  
print(f"Search tool description: {search_memory_tool.description}")
print(f"Search tool arguments: {search_memory_tool.args}")


# Agent System prompt that mentions the new memory tools
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings  
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}

IMPORTANT: Use memory tools to remember important information about people, projects, preferences, and past conversations.
Always search memory first to see if you have relevant context before responding to emails.
</ Instructions >
"""

# Function to create the system prompt for the main agent (same as before but with new prompt template)
def create_prompt(state):
    """Create the system prompt for the main agent including memory capabilities"""
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile  # This unpacks name, full_name, user_profile_background
            )
        }
    ] + state['messages']  # Add conversation history from the state

# Create list of tools for the agent - NOW INCLUDES MEMORY TOOLS!
tools = [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,    # NEW: Can store information
    search_memory_tool     # NEW: Can retrieve information
]

# Create the main ReAct agent that can use tools INCLUDING memory
response_agent = create_react_agent(
    "openai:gpt-4o",                      # Use GPT-4 instead of Claude
    tools=tools,                          # All available tools including memory
    prompt=create_prompt,                 # Function that creates the prompt
    store=store                          # IMPORTANT: Pass the memory store to the agent
)


# Configuration that specifies which user's memory to use
# In a real app, this would come from user authentication
config = {"configurable": {"langgraph_user_id": "John Doe"}}

print("\n=== TESTING MEMORY: STORING INFORMATION ===")
# Test 1: Store some information about a person. From the system prompt, the agent knows it can use memory for important facts.
response = response_agent.invoke(
    {"messages": [{"role": "user", "content": "Jim is my friend and he works at Google"}]},
    config=config  # This tells the system which user's memory to use
)

# Print what the agent did
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

print("\n=== TESTING MEMORY: RETRIEVING INFORMATION ===")
# Test 2: Ask about the person - should remember from memory
response = response_agent.invoke(
    {"messages": [{"role": "user", "content": "Who is Jim and where does he work?"}]},
    config=config  # Same user, so should access same memory
)

# Print what the agent did
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

# Let's examine what's stored in memory
print("\n=== EXAMINING MEMORY CONTENTS ===")
print("Available namespaces:", store.list_namespaces())
print("All memories:", store.search(('email_assistant', 'john_doe', 'collection')))
print("Searching for 'Jim':", store.search(('email_assistant', 'john_doe', 'collection'), query="jim"))


# State definition 
class State(TypedDict):
    email_input: dict  # The incoming email to process
    messages: Annotated[list, add_messages]  # Conversation history with add_messages merger

# Triage router function - classifies emails and decides what to do
def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """
    This function:
    1. Takes an email from the state
    2. Classifies it using the Router
    3. Decides what to do next (respond, ignore, or notify)
    """
    # Extract email parts from state
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Build the system prompt for classification by filling in the template
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )
    
    # Build the user prompt with email content
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    
    # Use the Router to classify the email
    result = llm_router.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    
    # Decide what to do based on classification
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    
    return Command(goto=goto, update=update)

# =============================================================================
# CREATE THE COMPLETE EMAIL AGENT WITH MEMORY
# =============================================================================

# Create the state graph (workflow) - same as before
email_agent = StateGraph(State)

# Add nodes to the graph
email_agent = email_agent.add_node("triage_router", triage_router)  # Classifies emails
email_agent = email_agent.add_node("response_agent", response_agent)  # Responds with memory

# Define the flow: START -> triage_router -> (response_agent OR END)
email_agent = email_agent.add_edge(START, "triage_router")

# Compile the graph - IMPORTANT: Pass the store so the agent can access memory
email_agent = email_agent.compile(store=store)

# Generate workflow visualization (optional)
try:
    png_data = email_agent.get_graph(xray=True).draw_mermaid_png()
    with open("email_workflow_with_memory.png", "wb") as f:
        f.write(png_data)
    print("Workflow saved as 'email_workflow_with_memory.png'")
except Exception as e:
    print(f"Could not generate workflow diagram: {e}")

# =============================================================================
# TEST THE COMPLETE SYSTEM WITH MEMORY
# =============================================================================

print("\n=== TESTING EMAIL AGENT WITH MEMORY ===")

# Test with an email from Alice about API documentation
email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>", 
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

# Run the email agent - it should store information about Alice and the API issue
response = email_agent.invoke(
    {"email_input": email_input},
    config=config  # Use Abdullah's user memory
)

print("\n=== FIRST EMAIL RESPONSE ===")
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

print("\n=== TESTING FOLLOW-UP EMAIL ===")

# Test with a follow-up email from Alice - should remember the previous conversation
email_input_followup = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Follow up", 
    "email_thread": """Hi John,

Any update on my previous ask?

Thanks!
Alice""",
}

# Run the email agent on the follow-up - should use memory to understand context
response = email_agent.invoke(
    {"email_input": email_input_followup}, 
    config=config  # Same user memory
)

print("\n=== FOLLOW-UP EMAIL RESPONSE ===")
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

print("\n=== FINAL MEMORY STATE ===")
print("All stored memories:", store.search(('email_assistant', 'john_doe', 'collection')))

print("\n=== SYSTEM COMPLETE ===")

"""
SUMMARY OF WHAT THIS MEMORY-ENABLED EMAIL AGENT DOES:

1. SETUP: Same as before - loads environment, defines profile and rules

2. MEMORY SYSTEM:
   - InMemoryStore: Local database with embedding-based search (like RAG)
   - manage_memory_tool: Lets agent store facts, preferences, context
   - search_memory_tool: Lets agent retrieve relevant information

3. ENHANCED AGENT:
   - Same triage and response capabilities as before
   - NEW: Can remember information across conversations
   - NEW: Searches memory before responding to emails

4. MEMORY WORKFLOW:
   - Agent automatically stores important information from emails
   - Agent searches memory when processing new emails  
   - Agent uses context from memory to provide better responses

5. TESTING:
   - First email: Agent learns about Alice and API documentation issue
   - Follow-up email: Agent remembers previous context and provides relevant response

The result is an intelligent email assistant that builds knowledge over time!

KEY DIFFERENCES FROM BASIC VERSION:
- Adds semantic memory (facts, people, projects, preferences)
- Agent can reference previous conversations and context
- Responses become more personalized and contextually aware
- Memory persists across different email processing sessions
"""

"""
So basically, this is all using RAG. 
Anytime user mentions something interesting or explicitly asks to add memory, agent has access to memory tool which is local database to store that info in a vector database (they technically store vectors). 
So in future interactions, agent always searches for the vector store for any relevant context. And in that store, it is organized by namespaces defined by userId so it is per user.
"""