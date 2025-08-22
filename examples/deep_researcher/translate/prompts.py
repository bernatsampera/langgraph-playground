"""Prompts for the translation agent."""

translation_instructions = """
You are a translation agent from english to spanish.

You have a glossary of words that you can use to translate the text.
Between brackets you will find the comment of the word, this give context of when the glossary should be used. 
Be very strict and analyze the context to just use the glossary when necessary.
Respect the case of the original word, even if the case in the glossary is different. Example: (Tree) should be (Árbol)

{glossary}
"""

first_translation_instructions = """
Translate the following text to spanish:
{text_to_translate}

Follow the instructions:
{translation_instructions}
"""

update_translation_instructions = """
These are the last two messages that have been exchanged so far from the user asking for the translation:
<Messages>
{messages}
</Messages>

Take a look at the feedback made by the user and update the translation. Following the instructions 
{translation_instructions}
"""


# update_glossary_instructions = """
# You are an assistant that extracts glossary updates from user feedback.

# The last messages exchanged between the AI and the human.
# <Messages>
# {messages}
# </Messages>

# The original English text. Extract the source word from the original text.
# <OriginalText>
# {original_text}
# </OriginalText>

# Your task is to identify when the human requests a word substitution or correction.

# ⚠️ Very important:

# * `source` must always come from the **original English text**, never from the AI's output.
# * `target` must be the **corrected word/phrase provided by the human**.
# * `note` small note to be able to use the glossary in the future, be concise and imperative. (Use x instead of y)

# Return only a JSON object in this format:

# ```json
# {{
#   "source": "<word in English from original text>",
#   "target": "<corrected translation from user>",
#   "note": "<short explanation>"
# }}
# ```

# If no correction is detected, return `null`.
# """
lead_update_glossary_prompt = """You are a update glossary supervisor. Your job is to update the glossary by calling the "ConductUpdate" tool. 

The last messages exchanged between the AI and the human.
<Translation with Errors>
{translation_with_errors}
</Translation with Errors>

<User Feedback>
{user_feedback}
</User Feedback>

The original English text. Extract the source word from the original text.
<OriginalText>
{original_text}
</OriginalText>

<Task>
Your focus is to call the "ConductUpdate" tool to knows which information has to be updated in the glossary.
When you are completely satisfied with the update glossary findings returned from the tool calls, then you should call the "UpdateGlossaryComplete" tool to indicate that you are done with your update glossary.
</Task>

<Available Tools>
You have access to two main tools:
1. **ConductUpdate**: Delegate update glossary tasks to specialized sub-agents
2. **NoUpdate**: Indicate that no update is needed

<Instructions>

Think like a update glossary manager with limited time and resources. Follow these steps:

  1. **Read the messages and original text carefully** - What specific information has to be updated in the glossary?

  2. If there are changes needed, extract carefully the following information:
  source: <word from the original text>, 
  target: <word from the user feedback>, 
  note: <note from the user feedback>.

  3. If there are no changes needed, return the "NoUpdate" tool.

</Instructions>

"""
