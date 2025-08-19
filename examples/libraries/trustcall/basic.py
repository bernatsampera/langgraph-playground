from langchain_google_genai import ChatGoogleGenerativeAI
import pprint
import json
from .schema import TelegramAndTrustFallPreferences
from dotenv import load_dotenv

load_dotenv()



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
bound = llm.bind_tools([TelegramAndTrustFallPreferences], 
                        tool_choice="TelegramAndTrustFallPreferences")

conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

result = bound.invoke(f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>""") 

# Pretty print the extracted preferences
print("=== Extracted Preferences ===")
if result.tool_calls: # type: ignore
    tool_call = result.tool_calls[0] #  type: ignore
    print(f"Tool: {tool_call['name']}")
    print("Preferences:")
    print(json.dumps(tool_call['args'], indent=2))