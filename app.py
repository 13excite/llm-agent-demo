import json
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

app = FastAPI(title="DevOps & Dev Multi-Agent Service")

llm = ChatOllama(model="llama3", temperature=0)


class TaskRequest(BaseModel):
    task: str

class AgentState(TypedDict):
    task: str
    specialist: str
    solution: str


def router_agent(state: AgentState):
    prompt = f"""You are the Senior Systems Architect. Your job is to classify incoming technical issues.
    
    If the issue is related to: Docker, CI/CD pipelines, GitHub Actions, Linux, Bash, Kubernetes, server logs, access rights, or deployment — return strictly the word: DEVOPS
    If the issue is related to: code (Python, JS, etc.), databases, SQL queries, API endpoint creation, or application logic bugs — return strictly the word: DEVELOPER
    
    Issue: {state['task']}
    Answer (only one word, DEVOPS or DEVELOPER):"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    
    # Clean possible artifacts from the model's response
    if "DEVOPS" in decision:
        specialist = "devops"
    else:
        specialist = "developer"
    return {"specialist": specialist}

def devops_agent(state: AgentState):
    prompt = f"""You are a Senior DevOps Engineer. Solve the automation problem or resolve the infrastructure incident.
    Provide a step-by-step solution, configuration files (if needed), or bash commands.
    
    Task: {state['task']}
    Solution:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content}

def developer_agent(state: AgentState):
    prompt = f"""You are a Senior Backend Developer. Fix the bug in the code, write a function, or optimize the query.
    Provide clean code with a brief explanation.
    
    Task: {state['task']}
    Solution:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content}

def route_decision(state: AgentState) -> Literal["devops", "developer"]:
    return state["specialist"]


workflow = StateGraph(AgentState)

workflow.add_node("router", router_agent)
workflow.add_node("devops", devops_agent)
workflow.add_node("developer", developer_agent)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "devops": "devops",
        "developer": "developer"
    }
)

workflow.add_edge("devops", END)
workflow.add_edge("developer", END)

agent_graph = workflow.compile()


@app.post("/api/agent/solve")
async def solve_task(request: TaskRequest):
    try:
        initial_state = {
            "task": request.task,
            "specialist": "",
            "solution": ""
        }
        
        result = agent_graph.invoke(initial_state)
        
        return {
            "status": "success",
            "assigned_to": result["specialist"],
            "task_received": request.task,
            "solution": result["solution"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

