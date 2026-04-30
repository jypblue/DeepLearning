from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, TypedDict, Dict
from contextlib import AsyncExitStack
import asyncio
import nest_asyncio
import os

nest_asyncio.apply()

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-chat"
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()


    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]

        # DeepSeek tool_choice: "auto" lets model decide; required when tools provided
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2024,
            tools=self.available_tools if self.available_tools else None,
            tool_choice="auto" if self.available_tools else None,
            messages=messages,
        )

        process_query = True
        while process_query:
            choice = response.choices[0]
            message = choice.message

            # Print text content if any
            if message.content:
                print(message.content)

            # Check if model wants to call tools
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                # Append assistant message with tool_calls
                messages.append({
                    'role': 'assistant',
                    'content': message.content,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    import json
                    tool_args = json.loads(tool_call.function.arguments)
                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # call the tool using the session   
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': result_text,
                    })

                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2024,
                    tools=self.available_tools,
                    tool_choice="auto",
                    messages=messages,
                )
            else:
                # No more tool calls — done
                if not message.content:
                    # content was already printed in a prior iteration
                    pass
                process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("MCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("\n")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "research_server.py"],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                response = await session.list_tools()
                tools = response.tools
                print("Connected to server with tools:", [tool.name for tool in tools])

                # Convert MCP tool schemas to OpenAI function-calling format
                self.available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                    }
                    for tool in tools
                ]

                await self.chat_loop()

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_server_and_run()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
