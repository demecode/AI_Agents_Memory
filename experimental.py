
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from typing import Literal
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Load environment variables from .env file (need OPENAI_API_KEY)
_ = load_dotenv()


# Basic user facts
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# Triage rules and agent instructions
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Create an in-memory store for BOTH semantic memory AND episodic memory
# This is the same local vector database, but we'll use different namespaces for different types of memory
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}  # Uses OpenAI's embedding model to create vectors
)

# =============================================================================
# SEED THE EPISODIC MEMORY - Add some example classifications to learn from
# =============================================================================

# Example 1: A question that should be RESPONDED to
example_1 = {
    "email": {
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
    },
    "label": "respond"  # This is what the agent should learn to do
}

# Store this example in the "examples" namespace (episodic memory)
# This is different from "collection" (semantic memory) - different types of memory in different folders
store.put(
    ("email_assistant", "john_doe", "examples"),  # namespace: app, user, memory_type
    str(uuid.uuid4()),                            # unique ID for this example
    example_1                                     # the actual example data
)

# Example 2: An informational email that should be IGNORED
example_2 = {
    "email": {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Update: Backend API Changes Deployed to Staging",
        "email_thread": """Hi John,

Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:

- Implemented JWT refresh token rotation
- Added rate limiting for login attempts
- Updated API documentation with new endpoints

All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*

No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.

Best regards,
Sarah""",
    },
    "label": "notify"  # This should be noted but not responded to
}

# Store the second example
store.put(
    ("email_assistant", "john_doe", "examples"),
    str(uuid.uuid4()),
    example_2
)

print("=== EPISODIC MEMORY SEEDED ===")
print("Stored 2 example email classifications for the agent to learn from")

# =============================================================================
# EPISODIC MEMORY HELPER FUNCTIONS - Format examples for the LLM
# =============================================================================

# Template for formatting an example to put in the prompt
# This shows the LLM: "Here's how you classified similar emails before"
example_template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

def format_few_shot_examples(examples):
    """
    Takes a list of example classifications and formats them for the LLM prompt.
    This is how we give the LLM "episodic memory" - examples of past decisions.
    """
    if not examples:
        return "No previous examples found."
    
    formatted_examples = ["Here are some previous examples:"]
    for example in examples:
        formatted_example = example_template.format(
            subject=example.value["email"]["subject"],
            to_email=example.value["email"]["to"],
            from_email=example.value["email"]["author"],
            content=example.value["email"]["email_thread"][:400],  # Truncate long emails
            result=example.value["label"],
        )
        formatted_examples.append(formatted_example)
    
    return "\n\n------------\n\n".join(formatted_examples)

# =============================================================================
# ENHANCED TRIAGE SYSTEM - Now uses episodic memory (few-shot examples)
# =============================================================================

# Router class for structured output (same as before)
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""
    
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(Router)

# UPDATED system prompt that includes few-shot examples from episodic memory
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above - learn from past decisions!

{examples}
</ Few shot examples >
"""

# User prompt template (same as before)
triage_user_prompt = """Please analyze this email and classify it:

FROM: {author}
TO: {to}
SUBJECT: {subject}

EMAIL CONTENT:
{email_thread}

Provide your classification with detailed reasoning."""

# =============================================================================
# STATE AND ENHANCED TRIAGE ROUTER - Now searches episodic memory for examples
# =============================================================================

# State definition (same as before)
class State(TypedDict):
    email_input: dict  # The incoming email to process
    messages: Annotated[list, add_messages]  # Conversation history

def triage_router(state: State, config, store) -> Command[Literal["response_agent", "__end__"]]:
    """
    Enhanced triage router that uses episodic memory (past examples) to classify emails.
    
    1. Extracts email details from state
    2. SEARCHES episodic memory for similar past classifications
    3. Uses those examples as few-shot prompts to improve classification
    4. Decides what to do next based on classification
    """
    # Extract email parts from state (same as before)
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # NEW: Search episodic memory for similar examples
    # This uses vector similarity search to find past email classifications that are similar to the current email
    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"  # This is the episodic memory namespace, different from "collection" (semantic memory)
    )
    
    # Search for similar examples using the current email as the query
    # The vector database will find past emails that are semantically similar
    similar_examples = store.search(
        namespace, 
        query=str({"email": state['email_input']}),  # Use current email to find similar past emails
        limit=3  # Get up to 3 most similar examples
    ) 
    
    # Format the examples for the LLM prompt
    formatted_examples = format_few_shot_examples(similar_examples)
    
    print(f"=== EPISODIC MEMORY SEARCH ===")
    print(f"Found {len(similar_examples)} similar examples from past classifications")
    
    # Build the system prompt with episodic examples included
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=formatted_examples  # NEW: Include past examples in the prompt
    )
    
    # Build the user prompt with email content
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    
    # Use the Router to classify the email (now with episodic memory context)
    result = llm_router.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    
    # Decide what to do based on classification (same logic as before)
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
# SAME TOOLS AS BEFORE - Actions the agent can take
# =============================================================================

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# Memory tools for semantic memory (facts about people, projects, etc.)
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"  # This is semantic memory (facts), different from "examples" (episodic memory)
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"  # This is semantic memory (facts)
    )
)

# =============================================================================
# MAIN AGENT WITH MEMORY - Same as before
# =============================================================================

# System prompt for the main response agent (same as before)
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

def create_prompt(state):
    """Create the system prompt for the main agent"""
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
            )
        }
    ] + state['messages']

# Create list of tools for the agent
tools = [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,    # Semantic memory (facts)
    search_memory_tool     # Semantic memory (facts)
]

# Create the main ReAct agent (same as before)
response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    store=store  # Pass the memory store
)

# =============================================================================
# CREATE THE COMPLETE EMAIL AGENT WITH BOTH MEMORY TYPES
# =============================================================================

# Create the state graph (workflow)
email_agent = StateGraph(State)

# Add nodes to the graph
email_agent = email_agent.add_node("triage_router", triage_router)  # Now uses episodic memory
email_agent = email_agent.add_node("response_agent", response_agent)  # Uses semantic memory

# Define the flow: START -> triage_router -> (response_agent OR END)
email_agent = email_agent.add_edge(START, "triage_router")

# Compile the graph with the memory store
email_agent = email_agent.compile(store=store)

# =============================================================================
# TEST THE EPISODIC MEMORY SYSTEM
# =============================================================================

# Configuration for a test user
config = {"configurable": {"langgraph_user_id": "john_doe"}}

print("\n=== TESTING EPISODIC MEMORY: SIMILAR TO TRAINING EXAMPLE ===")
# Test with an email similar to our training example (should be classified as "respond")
email_input_1 = {
    "author": "Bob Wilson <bob.wilson@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Question about REST API endpoints",
    "email_thread": """Hi John,

I'm working on the mobile app integration and have a question about the user authentication API. I noticed there might be some inconsistencies in the endpoint documentation.

Could you help clarify the correct endpoint for password reset?

Thanks!
Bob""",
}

response = email_agent.invoke({"email_input": email_input_1}, config=config)

print("\n=== ADDING A NEW EXAMPLE TO EPISODIC MEMORY ===")
# Simulate adding a new example - maybe the agent got it wrong and we want to correct it
spam_example = {
    "email": {
        "author": "Sales Team <sales@random-company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Incredible offer just for you!",
        "email_thread": """Hi John,

We have an amazing offer on premium software tools! Limited time only!

Buy now and save 80%! Don't miss out!

Click here: www.definitely-not-spam.com

Best regards,
Sales Team""",
    },
    "label": "ignore"  # This should definitely be ignored
}

# Add this as a new episodic memory example
store.put(
    ("email_assistant", "john_doe", "examples"),
    str(uuid.uuid4()),
    spam_example
)

print("Added spam example to episodic memory")

print("\n=== TESTING WITH SIMILAR SPAM EMAIL ===")
# Test with a similar spam email - should now be classified as "ignore" based on the example
spam_test = {
    "author": "Marketing Pro <marketing@super-deals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Amazing deal for software engineers!",
    "email_thread": """Hello John,

Special offer just for developers like you!

Get 90% off on our premium development suite!

Limited time - act now!

Visit: www.great-deals-totally-legit.com

Marketing Team""",
}

response = email_agent.invoke({"email_input": spam_test}, config=config)

print("\n=== EXAMINING EPISODIC MEMORY CONTENTS ===")
all_examples = store.search(("email_assistant", "john_doe", "examples"))
print(f"Total examples stored: {len(all_examples)}")
for i, example in enumerate(all_examples):
    print(f"Example {i+1}: {example.value['email']['subject']} -> {example.value['label']}")

print("\n=== SYSTEM COMPLETE ===")

"""
SUMMARY OF EPISODIC MEMORY SYSTEM:

1. EPISODIC MEMORY STORAGE:
   - Uses same vector database as semantic memory but different namespace ("examples" vs "collection")
   - Stores complete email + classification pairs as learning examples
   - Each example shows: "For this type of email, do this action"

2. FEW-SHOT LEARNING:
   - When classifying new emails, searches for similar past examples
   - Uses vector similarity to find relevant past decisions
   - Includes past examples in the LLM prompt as few-shot examples

3. CONTINUOUS LEARNING:
   - System can learn from corrections by adding new examples
   - Agent gets better at classification over time
   - Mistakes can be corrected by adding counter-examples

4. DUAL MEMORY SYSTEM:
   - Semantic memory ("collection"): Facts about people, projects, preferences
   - Episodic memory ("examples"): Past email classification decisions
   - Both use same vector database but different namespaces

5. IMPROVED CLASSIFICATION:
   - Agent now classifies based on rules + past examples
   - More accurate and consistent classification over time
   - Can adapt to user's specific preferences and patterns

The result is an email assistant that learns from experience and gets better over time!
"""

"""
So basically, instead of storing these few shot examples in the system prompt, literally a prompt engineering technique, we store them all in a vector database because we can store more of them
"""