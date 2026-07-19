# Simple LLM Agent Demo

This is a simple demo of an LLM agent based on LangChain that can solve
programming-related tasks. The agent is designed to receive a task description,
process it, and return a solution.

## Installation

To install the required dependencies, run the following command:

```shell
pip install -r requirements.txt
```

Then next, install `ollama`, run `serve` and pull `gemma4` model:

```shell
# ollama serve

# ollama pull gemma4

# ollama list
NAME             ID              SIZE      MODIFIED
gemma4:latest    c6eb396dbd59    9.6 GB    4 days ago
```

## Running the Agent

To start the agent, run the following command:

```shell
uvicorn app:app --reload
```

## Usage

Here is an example of how agent works with programming-related tasks:

```shell
# curl -X POST http://127.0.0.1:8000/api/agent/solve \
 -H "Content-Type: application/json" \
 -d '{"task": "I see error <ZeroDivisionError: division by zero> in my python"}'|jq .

{
  "status": "success",
  "assigned_to": "developer",
  "task_received": "I see error <ZeroDivisionError: division by zero> in my python",
  "solution": "Use an `if` statement to check if the divisor is zero before performing the division, or wrap the operation in a `try...except ZeroDivisionError` block."
}
```

Here is another example of how agent works with DevOps-related tasks:

```shell
# curl -X POST http://127.0.0.1:8000/api/agent/solve \
 -H "Content-Type: application/json" \
 -d '{"task": "My kubernetes pod keeps crash looping"}'|jq .

{
  "status": "success",
  "assigned_to": "devops",
  "task_received": "My kubernetes pod keeps crash looping",
  "solution": "`kubectl describe pod <pod-name> -n <namespace>` to check events, exit codes, and resource constraints."
}
```
