
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from data_tools import create_data_tools


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. "
        "Add your OpenRouter API key to the .env file."
    )


# --------------------------------------------------
# 2. Load the CSV dataset
# --------------------------------------------------
CSV_PATH = Path(__file__).parent / "sales.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_PATH}\n"
        "Place sales.csv in the project folder."
    )

try:
    df = pd.read_csv(CSV_PATH)
except Exception as error:
    raise RuntimeError(
        f"Failed to read the CSV file: {error}"
    ) from error

if df.empty:
    raise ValueError("The CSV file contains no data rows.")

if len(df.columns) == 0:
    raise ValueError("The CSV file contains no columns.")

print("\nDataset loaded successfully!")
print(f"File: {CSV_PATH.name}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# --------------------------------------------------
# 3. Connect to OpenRouter
# --------------------------------------------------
llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_retries=2,
    timeout=60,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "CSV Data Analysis Agent",
    },
)


# --------------------------------------------------
# 4. Create the data analysis tools
# --------------------------------------------------
tools = create_data_tools(df)


# --------------------------------------------------
# 5. Create the LangChain AI agent
# --------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a careful CSV Data Analysis Assistant.

    Your task is to answer questions about the loaded dataset.

    IMPORTANT RULES:

    1. Use the provided tools for numerical calculations.
    2. Never invent statistics or numerical results.
    3. Use exact column names from the dataset.
    4. For dataset overview questions, use inspect_dataset.
    5. For descriptive statistics, use summarize_column.
    6. For grouping and aggregation, use group_and_aggregate.
    7. For revenue questions, use revenue_by_group.
    8. Explain results in simple, clear language.
    9. If an analysis is unsupported, explain why.
    10. Never claim a calculation was performed unless
        the tool output supports the answer.
    11. If a requested column does not exist, explain
        the limitation.
    12. Distinguish original data from calculated values.
    13. For revenue, use quantity multiplied by unit_price
        through the revenue_by_group tool.
    14. If the user asks for revenue by region, group by
        the region column.
    15. If the user asks for revenue by product, group by
        the product column.
    16. If the user asks for revenue by month, first check
        whether the dataset contains a suitable date column.
        Do not invent dates or monthly values.
    17. If a tool returns an error, explain that error
        instead of fabricating a result.
    """,
)


# --------------------------------------------------
# 6. Start the interactive chat
# --------------------------------------------------
print("\nOpenRouter CSV Data Analysis Agent is ready!")
print("Ask questions about your dataset.")
print("Type 'exit' or 'quit' to stop.\n")


while True:
    try:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        # Find the final assistant response.
        messages = result["messages"]

        final_message = next(
            (
                message
                for message in reversed(messages)
                if getattr(message, "type", None) == "ai"
                and not getattr(message, "tool_calls", None)
            ),
            None,
        )

        if final_message is None:
            print("\nAgent: No final response was returned.\n")
            continue

        answer = final_message.content

        # Handle either string or block-based responses.
        if isinstance(answer, list):
            answer = "\n".join(
                block["text"]
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text")
            )

        print(f"\nAgent: {answer}\n")

    except KeyboardInterrupt:
        print("\n\nAgent stopped.")
        break

    except Exception as error:
        print(f"\nError: {error}")
        print(
            "Check your internet connection, OpenRouter API key, "
            "model availability, free-model limits, and tool support.\n"
        )
