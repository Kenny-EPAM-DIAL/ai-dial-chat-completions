import asyncio

from task.clients.client import DialClient
from task.constants import DEFAULT_SYSTEM_PROMPT
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


def strip_think(content: str) -> str:
    """
    Remove reasoning sections like <think>...</think> from model output.
    Keeps only the final assistant answer.
    """
    end_tag = "</think>"
    if end_tag in content:
        content = content.split(end_tag, 1)[1]
    return content.lstrip()


async def start(stream: bool) -> None:
    # -------------------------------
    # MODEL SELECTION
    # -------------------------------
    print("Select deployment model:")
    print("1) deepseek-r1  (recommended)")
    print("2) o3-mini-2025-01-31")
    print("3) o4-mini-2025-04-16")
    print("4) gpt-oss-120b")
    print("5) claude-3-5-sonnet-v2@20241022")
    print()

    choice = input("Enter number or type a custom deployment ID: ").strip()

    mapping = {
        "1": "deepseek-r1",
        "2": "o3-mini-2025-01-31",
        "3": "o4-mini-2025-04-16",
        "4": "gpt-oss-120b",
        "5": "claude-3-5-sonnet-v2@20241022",
    }

    deployment_name = mapping.get(choice, choice or "deepseek-r1")
    print(f"\nUsing deployment: {deployment_name}\n")

    # -------------------------------
    # CLIENT INITIALIZATION
    # -------------------------------
    client = DialClient(deployment_name)

    # -------------------------------
    # CONVERSATION SETUP
    # -------------------------------
    conversation = Conversation()

    print("Provide System prompt or press Enter to use default.")
    system_prompt = input("> ").strip() or DEFAULT_SYSTEM_PROMPT
    conversation.add_message(Message(Role.SYSTEM, system_prompt))

    print("\nType your question or 'exit' to quit.\n")

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    while True:
        user_input = input("> ").strip()

        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        if not user_input:
            continue

        conversation.add_message(Message(Role.USER, user_input))

        # -------------------------------
        # CHAT COMPLETION
        # -------------------------------
        if stream:
            # DialClient.stream_completion already prints tokens as they arrive.
            print("AI: ", end="", flush=True)
            ai_msg = await client.stream_completion(conversation.get_messages())
            # Clean content for history (even though streaming already showed raw text)
            ai_msg.content = strip_think(ai_msg.content)
        else:
            ai_msg = client.get_completion(conversation.get_messages())
            clean = strip_think(ai_msg.content)
            ai_msg.content = clean
            print(f"AI: {clean}")

        # Save assistant message in history
        conversation.add_message(ai_msg)


if __name__ == "__main__":
    asyncio.run(start(True))
