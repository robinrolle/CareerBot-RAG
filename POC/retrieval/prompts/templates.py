
# Prompt Template for the Final Answer
MY_PROMPT_TEMPLATE = """
### TASK
You are doing work skills or occupation matching. You are given a list of items from which you need to pick those that match best with a given text content. 

### RULES OF TASK
Strictly Follow these guidelines when answering:
1. You can only choose items from a given list below. DO NOT INVENT any new items not in list.
2. For chosen skills or occupations, give a scoring of LOW, MEDIUM or HIGH depending how well each item matches the content of the user text.
3. Give your response in JSON format {{'items':[item1, item2,...],'relevances':['LOW','MEDIUM',...]}}

### BEGIN USER TEXT 

{question}

### END USER TEXT

The following list contains skills or occupations with their descriptions. Each ITEM and DESCRIPTION are separated by a pipe "|" symbol. Only include items in your response.
Think carefully and choose maximum {LLM_MAX_PICKS} items from the list, remember RULES OF TASK.

### BEGIN ITEM DESCRIPTIONS

{context}

### END ITEM DESCRIPTIONS

Your answer:
"""