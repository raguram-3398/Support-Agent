import argparse
import asyncio

from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError

from support_agent.graph import MAX_STEPS, build_graph, make_model


async def main(customer_id: int, question: str) -> None:
    graph = build_graph(make_model())
    config = {"recursion_limit": MAX_STEPS, "callbacks": [CallbackHandler()]}
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(question)], "customer_id": customer_id},
            config,
        )
        print(result["messages"][-1].text)
    except GraphRecursionError:
        print(f"Stopped: step cap of {MAX_STEPS} reached.")
    finally:
        get_client().flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", type=int, required=True)
    parser.add_argument("question")
    args = parser.parse_args()
    asyncio.run(main(args.customer, args.question))
