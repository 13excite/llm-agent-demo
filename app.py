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
    prompt = f"""Ты — Главный системный архитектор. Твоя задача — классифицировать входящую техническую проблему.
    
    Если проблема связана с: Docker, CI/CD pipelines, GitHub Actions, Linux, Bash, Kubernetes, логами серверов, правами доступа или деплоем — верни строго слово: DEVOPS
    Если проблема связана с: кодом (Python, JS и т.д.), базами данных, SQL-запросами, созданием API-эндпоинтов или багами в логике приложения — верни строго слово: DEVELOPER
    
    Проблема: {state['task']}
    Ответ (только одно слово, DEVOPS или DEVELOPER):"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    
    # Очистка от возможных артефактов модели
    if "DEVOPS" in decision:
        specialist = "devops"
    else:
        specialist = "developer"
        
    return {"specialist": specialist}

def devops_agent(state: AgentState):
    prompt = f"""Ты — Senior DevOps Engineer. Реши проблему автоматизации или устрани инцидент в инфраструктуре.
    Предоставь пошаговое решение, конфигурационные файлы (если нужны) или bash-команды.
    
    Задача: {state['task']}
    Решение:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content}

def developer_agent(state: AgentState):
    prompt = f"""Ты — Senior Backend Developer. Исправь баг в коде, напиши функцию или оптимизируй запрос.
    Предоставь чистый код с кратким объяснением.
    
    Задача: {state['task']}
    Решение:"""
    
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

