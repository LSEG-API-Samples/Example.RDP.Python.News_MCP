import pytest
import json
import os
import sys
import asyncio
import logging
from typing import List, Dict, Any

from langsmith import testing as t
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
from langchain_core.messages import HumanMessage

# Add parent directory to path to import chat_app and llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat_app import initialize_app
from llm import get_default_judge_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

judge_llm = get_default_judge_llm()

trajectory_evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    judge=judge_llm,
    continuous=True,
    feedback_key="news_chat_trajectory_accuracy",
)


async def run_agent_get_trajectory(question: str) -> List[Dict[str, Any]]:
    """Run the agent and get the trajectory for a given question."""
    try:
        # Initialize the app if not already done
        await initialize_app()
        from chat_app import graph

        if graph is None:
            raise Exception("Graph is not initialized.")

        # Create initial state with user message
        initial_state = {"messages": [HumanMessage(content=question)]}

        # Run the graph
        result = await graph.ainvoke(initial_state)

        # Convert LangChain messages to standard chat format
        trajectory = []
        for message in result["messages"]:
            if hasattr(message, "type"):
                if message.type == "human":
                    trajectory.append({"role": "user", "content": message.content})
                elif message.type == "ai":
                    msg_dict = {"role": "assistant", "content": message.content or ""}
                    # Handle tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        msg_dict["tool_calls"] = []
                        for tool_call in message.tool_calls:
                            msg_dict["tool_calls"].append(
                                {
                                    "function": {
                                        "name": tool_call.get("name", ""),
                                        "arguments": json.dumps(
                                            tool_call.get("args", {})
                                        ),
                                    }
                                }
                            )
                    trajectory.append(msg_dict)
                elif message.type == "tool":
                    trajectory.append({"role": "tool", "content": message.content})

        return trajectory

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return []


@pytest.mark.langsmith
def test_trajectory_accuracy_query_news():
    """Test trajectory accuracy for a query on a companies latest news"""
    # Run the async function to get the actual trajectory
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        outputs = loop.run_until_complete(
            run_agent_get_trajectory("What is happening with Tesla")
        )
    finally:
        loop.close()

    # Reference trajectory for comparison
    reference_outputs = [
        {"role": "user", "content": "What is happening with Tesla"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_headlines",
                        "arguments": json.dumps({"query": "Tesla"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "Latest Tesla news and developments..."},
        {"role": "assistant", "content": "Here's what's happening with Tesla..."},
    ]

    # Log to LangSmith
    t.log_inputs({"query": "What is happening with Tesla"})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    # Run the trajectory evaluator
    trajectory_evaluator(outputs=outputs, reference_outputs=reference_outputs)


@pytest.mark.langsmith
def test_trajectory_accuracy_no_tools():
    """Test trajectory accuracy for a query that does not use tools"""
    # Run the async function to get the actual trajectory
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        outputs = loop.run_until_complete(
            run_agent_get_trajectory("Hello, how are you?")
        )
    finally:
        loop.close()

    # Reference trajectory for comparison
    reference_outputs = [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you!"
        }
    ]

    # Log to LangSmith
    t.log_inputs({"query": "Hello, how are you?"})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    # Run the trajectory evaluator
    trajectory_evaluator(outputs=outputs, reference_outputs=reference_outputs)
