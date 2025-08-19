import json
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
from langmem import create_manage_memory_tool, create_search_memory_tool, create_multi_prompt_optimizer

# Load environment variables from .env file (need OPENAI_API_KEY and ANTHROPIC_API_KEY)
_ = load_dotenv()

# Basic user facts
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# DEFAULT triage rules and agent instructions - these will now be stored in procedural memory and can be updated
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}



# Create the same in-memory store, but now we'll use it for THREE types of memory:
# 1. "collection" namespace = semantic memory (facts about people, projects)
# 2. "examples" namespace = episodic memory (past email classification examples)  
# 3. User-specific namespace = procedural memory (system prompts and instructions)
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

print("=== PROCEDURAL MEMORY SYSTEM ===")
print("Procedural memory stores system prompts and instructions that can be dynamically updated")
print("This allows the agent to learn new procedures and modify its behavior over time")

# =============================================================================
# EPISODIC MEMORY HELPERS - Same as before
# =============================================================================

# Template for formatting episodic examples (same as before)
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

def format_few_shot_examples(examples):
    """Format episodic examples for few-shot learning in prompts"""
    if not examples:
        return "No previous examples found."
    
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


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

llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)

# Triage system prompt template (same as before)
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
Follow these examples more than any instructions above

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


# State definition (same as before)
class State(TypedDict):
    email_input: dict  # The incoming email to process
    messages: Annotated[list, add_messages]  # Conversation history

def triage_router(state: State, config, store) -> Command[Literal["response_agent", "__end__"]]:
    """
    Enhanced triage router that uses ALL THREE types of memory:
    1. Episodic memory - finds similar past classification examples
    2. Procedural memory - loads current triage rules from memory store
    3. Semantic memory - (used by response agent, not triage)
    
    MAIN CHANGE: Instead of using hardcoded triage rules, we now load them from procedural memory
    """
    # Extract email parts from state (same as before)
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Get episodic memory examples (same as before)
    episodic_namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(episodic_namespace, query=str({"email": state['email_input']})) 
    formatted_examples = format_few_shot_examples(examples)

    # NEW: Get procedural memory - load triage rules from memory store instead of hardcoded values
    user_id = config['configurable']['langgraph_user_id']
    procedural_namespace = (user_id,)  # Simple namespace for procedural memory

    # Load "ignore" rules from procedural memory
    result = store.get(procedural_namespace, "triage_ignore")
    if result is None:
        # First time - store the default rules in procedural memory
        store.put(
            procedural_namespace, 
            "triage_ignore", 
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
        print("📝 Initialized 'ignore' rules in procedural memory")
    else:
        # Load existing rules from procedural memory
        ignore_prompt = result.value['prompt']
        print(f"📚 Loaded 'ignore' rules from procedural memory: {ignore_prompt}")

    # Load "notify" rules from procedural memory
    result = store.get(procedural_namespace, "triage_notify")
    if result is None:
        store.put(
            procedural_namespace, 
            "triage_notify", 
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
        print("📝 Initialized 'notify' rules in procedural memory")
    else:
        notify_prompt = result.value['prompt']
        print(f"📚 Loaded 'notify' rules from procedural memory: {notify_prompt}")

    # Load "respond" rules from procedural memory
    result = store.get(procedural_namespace, "triage_respond")
    if result is None:
        store.put(
            procedural_namespace, 
            "triage_respond", 
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
        print("📝 Initialized 'respond' rules in procedural memory")
    else:
        respond_prompt = result.value['prompt']
        print(f"📚 Loaded 'respond' rules from procedural memory: {respond_prompt}")
    
    # Build the system prompt using prompts from procedural memory
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,      # From procedural memory, not hardcoded
        triage_notify=notify_prompt,  # From procedural memory, not hardcoded
        triage_email=respond_prompt,  # From procedural memory, not hardcoded
        examples=formatted_examples   # From episodic memory
    )
    
    # Build user prompt and classify (same as before)
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    
    result = llm_router.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    
    # Route based on classification (same logic as before)
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

# Memory tools for semantic memory (same as before)
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"  # Semantic memory namespace
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"  # Semantic memory namespace
    )
)

# =============================================================================
# ENHANCED MAIN AGENT - Now loads instructions from procedural memory
# =============================================================================

# System prompt template for main agent (same as before)
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
</ Instructions >
"""

def create_prompt(state, config, store):
    """
    Enhanced create_prompt function that loads agent instructions from procedural memory.
    
    MAIN CHANGE: Instead of using hardcoded instructions, we now load them from the memory store.
    This allows the instructions to be updated dynamically without changing code.
    """
    user_id = config['configurable']['langgraph_user_id']
    procedural_namespace = (user_id,)
    
    # Load agent instructions from procedural memory
    result = store.get(procedural_namespace, "agent_instructions")
    if result is None:
        # First time - store the default instructions in procedural memory
        store.put(
            procedural_namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        instructions_prompt = prompt_instructions["agent_instructions"]
        print("📝 Initialized agent instructions in procedural memory")
    else:
        # Load existing instructions from procedural memory
        instructions_prompt = result.value['prompt']
        print(f"📚 Loaded agent instructions from procedural memory: {instructions_prompt}")
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=instructions_prompt,  # From procedural memory, not hardcoded
                **profile
            )
        }
    ] + state['messages']

# Create list of tools and main agent (same as before)
tools = [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,    # Semantic memory
    search_memory_tool     # Semantic memory
]

response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,  # Now loads from procedural memory
    store=store
)

# =============================================================================
# CREATE THE COMPLETE EMAIL AGENT WITH ALL THREE MEMORY TYPES
# =============================================================================

# Create and compile the email agent (same as before)
email_agent = StateGraph(State)
email_agent = email_agent.add_node("triage_router", triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)

# =============================================================================
# TEST THE PROCEDURAL MEMORY SYSTEM
# =============================================================================

# Configuration for a test user
config = {"configurable": {"langgraph_user_id": "john_doe"}}

print("\n=== TESTING INITIAL BEHAVIOR ===")
# Test with an urgent email
email_input = {
    "author": "Alice Jones <alice.jones@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Urgent: Service Down",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why?

Thanks,
Alice""",
}

response = email_agent.invoke({"email_input": email_input}, config=config)

print("\n=== INITIAL EMAIL RESPONSE ===")
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

# Examine current procedural memory values
print("\n=== CURRENT PROCEDURAL MEMORY ===")
print("Agent instructions:", store.get(("john_doe",), "agent_instructions").value['prompt'])
print("Triage ignore rules:", store.get(("john_doe",), "triage_ignore").value['prompt'])
print("Triage notify rules:", store.get(("john_doe",), "triage_notify").value['prompt'])
print("Triage respond rules:", store.get(("john_doe",), "triage_respond").value['prompt'])

# =============================================================================
# PROCEDURAL MEMORY OPTIMIZATION - Update prompts based on feedback
# =============================================================================

print("\n=== PROCEDURAL MEMORY OPTIMIZATION ===")
print("Using LLM to update system prompts based on human feedback")

# Define feedback about how the agent should behave
conversations = [
    (
        response['messages'],
        "Always sign your emails 'John Doe'"  # Human feedback to improve email writing
    )
]

# Define the current prompts that can be updated
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("john_doe",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("john_doe",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"
    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("john_doe",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"
    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("john_doe",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"
    },
]

# Use LangMem's prompt optimizer to automatically update the prompts based on feedback
# This is an LLM that analyzes conversations and feedback to improve system prompts
optimizer = create_multi_prompt_optimizer(
    "openai:gpt-4o",  # Using GPT-4 instead of Claude
    kind="prompt_memory",
)

print("🔄 Running prompt optimization based on feedback...")
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)

print("Updated prompts:")
print(json.dumps(updated, indent=2))

# Update the procedural memory with the optimized prompts
print("\n=== UPDATING PROCEDURAL MEMORY ===")
for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"📝 Updated {name} prompt in procedural memory")
        
        if name == "main_agent":
            store.put(
                ("john_doe",),
                "agent_instructions",
                {"prompt": updated_prompt['prompt']}
            )
        elif name == "triage-ignore":
            store.put(
                ("john_doe",),
                "triage_ignore",
                {"prompt": updated_prompt['prompt']}
            )
        elif name == "triage-notify":
            store.put(
                ("john_doe",),
                "triage_notify",
                {"prompt": updated_prompt['prompt']}
            )
        elif name == "triage-respond":
            store.put(
                ("john_doe",),
                "triage_respond",
                {"prompt": updated_prompt['prompt']}
            )

print("\n=== TESTING UPDATED BEHAVIOR ===")
# Test the same email again to see if behavior improved
response = email_agent.invoke({"email_input": email_input}, config=config)

print("=== UPDATED EMAIL RESPONSE ===")
for m in response["messages"]:
    print(f"{m.type.upper()}: {m.content}")

print("\n=== TESTING ANOTHER PROCEDURAL UPDATE ===")
# Test updating triage rules to ignore emails from specific people
conversations_2 = [
    (
        response['messages'],
        "Ignore any emails from Alice Jones"  # Feedback to update triage rules
    )
]

updated_2 = optimizer.invoke(
    {"trajectories": conversations_2, "prompts": prompts}
)

# Update the ignore rules specifically
for i, updated_prompt in enumerate(updated_2):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        if name == "triage-ignore":
            print(f"📝 Updated ignore rules: {updated_prompt['prompt']}")
            store.put(
                ("john_doe",),
                "triage_ignore",
                {"prompt": updated_prompt['prompt']}
            )

print("\n=== TESTING UPDATED TRIAGE BEHAVIOR ===")
# Test the same email again - should now be ignored
response = email_agent.invoke({"email_input": email_input}, config=config)

print("\n=== FINAL PROCEDURAL MEMORY STATE ===")
print("Updated agent instructions:", store.get(("john_doe",), "agent_instructions").value['prompt'])
print("Updated ignore rules:", store.get(("john_doe",), "triage_ignore").value['prompt'])

print("\n=== SYSTEM COMPLETE ===")

"""
SUMMARY OF PROCEDURAL MEMORY SYSTEM:

1. PROCEDURAL MEMORY STORAGE:
   - Uses same vector database but user-specific namespace ("user_id" instead of "collection" or "examples")
   - Stores system prompts and instructions that define HOW the agent operates
   - Can be dynamically updated without changing code

2. DYNAMIC PROMPT LOADING:
   - Triage router loads classification rules from procedural memory
   - Main agent loads instructions from procedural memory
   - Falls back to defaults if nothing stored yet

3. AUTOMATED PROMPT OPTIMIZATION:
   - Uses LangMem's prompt optimizer to improve prompts based on feedback
   - Analyzes conversations and human feedback to suggest better instructions
   - Updates procedural memory automatically

4. TRIPLE MEMORY SYSTEM:
   - Semantic memory ("collection"): Facts about people, projects, preferences
   - Episodic memory ("examples"): Past email classification decisions  
   - Procedural memory ("user_id"): System prompts and instructions

5. CONTINUOUS IMPROVEMENT:
   - Agent can learn new procedures through prompt optimization
   - Behavior can be refined without code changes
   - Instructions evolve based on real usage and feedback

The result is an email assistant that can modify its own behavior and learn new procedures over time!

MAIN CHANGES FOR PROCEDURAL MEMORY:
1. triage_router() now loads rules from store instead of hardcoded values
2. create_prompt() now loads instructions from store instead of hardcoded values  
3. Added prompt optimization system to update stored prompts based on feedback
4. Uses user-specific namespace for procedural memory storage
"""


"""
Instead of hardcoding system prompts in the code, we store them in the same vector database but in a user-specific namespace. The agent loads its instructions, triage rules, and behavior guidelines from the database at runtime.
When we want the agent to learn new procedures or change its behavior, we don't modify code - we update the stored prompts in the database. The agent automatically uses the updated instructions next time it runs.
"""