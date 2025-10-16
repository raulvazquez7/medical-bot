"""
Main entry point for the medical chatbot LangGraph agent.
Compatible with LangGraph Studio and console execution.
Uses async/await for all LLM operations.
"""

import uuid
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from src import config
from src.graph.builder import build_graph
from src.utils.logger import configure_logging, get_logger, set_correlation_id

# Configure structured logging
configure_logging(level="INFO", use_structured=False)  # Use simple logs for console

logger = get_logger(__name__)

logger.info("graph_build_started", mode="studio")
app = build_graph(with_checkpointer=False)
logger.info("graph_build_completed", mode="studio")

console_app = build_graph(with_checkpointer=True)
logger.info("graph_build_completed", mode="console", checkpointer=True)


async def run_console_chat():
    """Async main loop for console chat interaction."""
    logger.info("console_mode_started")
    config.check_env_vars()

    logger.info("graph_visualization_started")
    try:
        png_bytes = console_app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        logger.info("graph_visualization_saved", file="graph.png")
    except Exception as e:
        logger.warning("graph_visualization_failed", error=str(e))

    logger.info("medical_bot_initialized", version="async_production")
    thread_id = str(uuid.uuid4())
    set_correlation_id(thread_id)
    logger.info("conversation_started", thread_id=thread_id)

    run_config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("Medical Chatbot (Async) - Type 'exit' or 'quit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() in ["exit", "quit"]:
                break

            inputs = {"messages": [HumanMessage(content=question)]}

            print("\n--- Processing... ---")
            async for event in console_app.astream(
                inputs, config=run_config, stream_mode="values"
            ):
                if "intent" in event and event["intent"]:
                    print(f"-> Intent: {event['intent']}")

                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    print(f"\nBot Response:\n{last_message.content}")

        except KeyboardInterrupt:
            logger.info("conversation_interrupted_by_user")
            break
        except Exception as e:
            logger.error("conversation_error", exc_info=True, error=str(e))
            print(f"\nError: {e}")
            break

    logger.info("conversation_ended", thread_id=thread_id)
    print("\nGoodbye! Take care.")


if __name__ == "__main__":
    asyncio.run(run_console_chat())
