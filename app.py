import streamlit as st
from typing import List, TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import json
import os 

load_dotenv()

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Essay Agent", page_icon="📝", layout="centered")
st.title("📝 Autonomous Essay Writing Agent")
st.markdown("Enter a topic, and watch the agent research, draft, critique, and polish a final essay.")

def get_groq_api_key() -> str | None:
    return (
        st.secrets.get("groq_api")
        or os.getenv("groq_api")
        or os.getenv("groq_api")
    )


def run_search(query: str, max_results: int = 5) -> str:
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results returned."

    formatted_results = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        link = item.get("href", "No link available")
        snippet = item.get("body", "No summary available")
        formatted_results.append(
            f"{index}. {title}\nURL: {link}\nSnippet: {snippet}"
        )
    return "\n\n".join(formatted_results)


groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error("Missing Groq API key. Add `groq_api` to Streamlit secrets or environment variables.")
    st.stop()

llm = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    api_key = groq_api_key,
    temperature = 0
)

# Define State
class AgentState(TypedDict):
    topic: str
    search_terms: List[str]
    research_data: str
    first_draft: str
    critique: str
    final_draft: str

# --- 2. AGENT NODES (Exactly as built in Colab) ---
def generate_keywords(state: AgentState):
    print("--- 1. GENERATING KEYWORDS ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research librarian and analytical planner. Your task is to take a broad essay topic and generate exactly 4 highly targeted search engine queries. 
        
        These queries must explore the topic from different, distinct angles to ensure a comprehensive foundation for an essay.
        
        CRITICAL INSTRUCTION: You must output the queries strictly as a JSON array of strings. Do not include any introductory text, conversational filler, or markdown formatting (like ```json). 
        Example output format: ["query 1", "query 2", "query 3", "query 4"]"""),
        ("human", "Topic: {topic}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({"topic": state["topic"]})

    try:
        raw_output = result.content.strip().replace('```json', '').replace('```', '')
        terms = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Falling back to default search.")
        terms = [state["topic"]] 
    print(terms)  
    return {"search_terms": terms}

def web_search(state: AgentState):
    print("--- 2. CONDUCTING WEB SEARCH & SYNTHESIZING ---")
    
    raw_results = []
    for term in state["search_terms"]:
        print(f"Searching for: {term}")
        try:
            result = run_search(term)
            raw_results.append(f"Source Query: '{term}'\nResults:\n{result}\n")
        except Exception as e:
            print(f"Search failed for {term}: {e}")
            
    combined_raw_data = "\n".join(raw_results)
    
    print("Synthesizing raw data...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Research Analyst. Your job is to review raw web search results and extract the most valuable, factual, and up-to-date information for an essay.
        
        Your specific tasks:
        1. Eliminate all noise, irrelevant links, and duplicate information.
        2. Extract key statistics, definitions, and core arguments.
        3. Prioritize Recency: Explicitly highlight any 'latest research', current trends, or recent developments mentioned in the data.
        4. Organize the extracted information into logical, categorized bullet points so the writer can easily use it.
        
        Do not write the essay. Only output the structured research notes."""),
        ("human", "Topic: {topic}\n\nRaw Search Results:\n{raw_data}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": state["topic"], 
        "raw_data": combined_raw_data
    })
    
    return {"research_data": result.content}

def write_first_draft(state: AgentState):
    print("--- 3. WRITING FIRST DRAFT ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert, highly skilled essay writer. Your task is to write a comprehensive, well-structured first draft on the provided Topic.
        
        CRITICAL INSTRUCTIONS:
        1. Strict Grounding: You must base your essay strictly and exclusively on the provided 'Research Notes'. Do not introduce external facts, statistics, historical dates, or claims that are not explicitly present in the provided research data. If the research data is sparse, write a shorter essay rather than making things up.
        2. Structure: Your draft must include an engaging introduction with a clear thesis statement, multiple well-organized body paragraphs, and a strong conclusion.
        3. Narrative Flow: Do not just list the facts. Seamlessly weave the categorized research points together. Use proper transitional phrases between paragraphs and ideas so the text reads like a cohesive, compelling narrative.
        4. Tone: Maintain a professional, objective, and analytical tone appropriate for a high-quality academic or professional essay."""),
        ("human", "Original Topic: {topic}\n\nResearch Notes from Analyst:\n{research_data}\n\nWrite the first draft now:")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": state["topic"], 
        "research_data": state["research_data"]
    })
    
    return {"first_draft": result.content}

def review_draft(state: AgentState):
    print("--- 4. REVIEWING DRAFT & FACT-CHECKING ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a ruthless, highly critical Academic Editor. Your job is to review a first draft of an essay and provide a step-by-step, actionable critique.
        
        Evaluate the draft strictly against this rubric:
        1. Factual Alignment (CRITICAL): Cross-reference the draft with the Research Notes. Explicitly flag any claims, statistics, or historical facts in the draft that are NOT present in the notes (hallucinations). Flag if crucial research points were ignored.
        2. Structure: Does the essay have a clear thesis statement, distinct body paragraphs, and a definitive conclusion? 
        3. Flow & Cohesion: Are the transitions between ideas smooth? Point out specific paragraphs or sentences that feel disconnected or abrupt.
        4. Tone: Ensure the tone remains consistently analytical and professional.
        
        OUTPUT REQUIREMENT:
        Provide your critique as a structured, bulleted list of ACTIONABLE directives for the final writer. Do not rewrite the essay yourself. Tell the writer exactly what to fix."""),
        ("human", "Original Topic: {topic}\n\nVerified Research Notes:\n{research_data}\n\nFirst Draft to Review:\n{first_draft}\n\nProvide your step-by-step critique now:")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": state["topic"], 
        "research_data": state["research_data"],
        "first_draft": state["first_draft"]
    })
    
    return {"critique": result.content}

def write_final_draft(state: AgentState):
    print("--- 5. WRITING FINAL DRAFT ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Master Essayist and the final author of this piece. Your job is to take a first draft and flawlessly integrate the feedback provided by the Academic Editor.
        
        CRITICAL INSTRUCTIONS:
        1. Strict Execution: You must implement every single actionable directive listed in the Editor's Critique. If the editor says to remove a sentence, remove it. If they say to combine paragraphs, combine them.
        2. Prose Refinement: Elevate the vocabulary, ensure perfect grammar, and polish the transitions so the essay reads beautifully.
        3. Output Formatting (ABSOLUTE RULE): Do NOT include any introductory text, pleasantries, or acknowledgments of the critique. Do not say 'Here is the revised draft'. Your output must be ONLY the title and the final text of the essay itself."""),
        ("human", "Original Topic: {topic}\n\nFirst Draft:\n{first_draft}\n\nEditor's Critique:\n{critique}\n\nWrite the final, polished essay now:")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "topic": state["topic"], 
        "first_draft": state["first_draft"], 
        "critique": state["critique"]
    })
    
    return {"final_draft": result.content}

# --- 3. GRAPH COMPILATION ---
# Cache the graph so it doesn't rebuild on every UI click
@st.cache_resource
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("keyword_generator", generate_keywords)
    workflow.add_node("web_researcher", web_search)
    workflow.add_node("first_drafter", write_first_draft)
    workflow.add_node("reviewer", review_draft)
    workflow.add_node("final_writer", write_final_draft)

    workflow.set_entry_point("keyword_generator")
    workflow.add_edge("keyword_generator", "web_researcher")
    workflow.add_edge("web_researcher", "first_drafter")
    workflow.add_edge("first_drafter", "reviewer")
    workflow.add_edge("reviewer", "final_writer")
    workflow.add_edge("final_writer", END)
    
    return workflow.compile()

agent_app = build_agent()

# --- 4. STREAMLIT UI & EXECUTION ---
topic = st.text_input("What should the essay be about?", placeholder="e.g., The evolution of Agentic AI...")

if st.button("Generate Essay", type="primary"):
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        # Use st.status to show the user what node is currently running
        with st.status("Initializing Agent Workflows...", expanded=True) as status_box:
            final_state = None
            
            # Stream the graph execution
            for event in agent_app.stream({"topic": topic}):
                for node_name, state_update in event.items():
                    # Format the node name for the UI (e.g., 'web_researcher' -> 'Web Researcher')
                    formatted_name = node_name.replace("_", " ").title()
                    st.write(f"✅ Completed step: **{formatted_name}**")
                    final_state = state_update # Keep track of the latest state
            
            status_box.update(label="Essay successfully generated!", state="complete", expanded=False)
        
        # Display the final output
        st.divider()
        st.subheader("Final Draft")
        st.write(final_state["final_draft"])
        
        # Optional: Add an expander to let the user see the agent's internal workings
        with st.expander("🔍 View Agent's Internal Research & Critique"):
            st.markdown("**Research Notes:**")
            st.write(final_state["research_data"])
            st.markdown("**Editor's Critique:**")
            st.write(final_state["critique"])
