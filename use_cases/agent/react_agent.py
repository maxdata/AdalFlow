#!/usr/bin/env python
# coding: utf-8

# # ReAct Agent Use Case

# # 1. Q&A Chatbot
# In this tutorial, we will implement ``adalflow ReAct`` to build a Q&A chatbot on [HotpotQA](https://arxiv.org/pdf/1809.09600) dataset. 
# 
# To learn more about ``adalflow ReAct``, please refer to our developer notes.

# 
# # 2. HotpotQA Dataset
# We are using [HotpotQA](https://arxiv.org/pdf/1809.09600). It is a Wikipedia-based multi-hop question and answer dataset.



# load the dataset
from datasets import load_dataset
dataset = load_dataset(path="hotpot_qa", name="fullwiki")




# check the data sample
test_sample = dataset["validation"][0]
print(f"len of eval: {len(dataset['validation'])}")
print(f"example: {test_sample}")
print(f"attributes in each sample: {list(test_sample.keys())}")




# Each sample contains a question and a corresponding answer.
print(f"question: {test_sample.get('question')}")
print(f"answer: {test_sample.get('answer')}")


# # 3. Set up
# Please make sure you have set the model client APIs before running the agent. Now import the necessary packages.



import dotenv
from adalflow.components.model_client import OpenAIClient
from adalflow.components.agent.react_agent import ReActAgent
from adalflow.core.tool_helper import FunctionTool

import time

# load evironment, please set the relative path to your .env file that includes the api key
dotenv.load_dotenv(dotenv_path="../../.env", override=True)


# # 4. Create Agent
# To create an gent, we need to define the basic components.
# 
# ## Tools
# Firstly, we need to specify what functions the agent will need to answer the question. In this case, we are answering the Wikipedia-based questions, we will allow the agent to **search** Wikipedia api. The [ReAct Paper](https://arxiv.org/pdf/2210.03629) includes a **lookup** function that serves as Ctrl+F functionality on the browser.
# 
# As ``adalflow ReAct`` has a built in ``finish`` function, we don't need to define by ourselves.



import requests
from bs4 import BeautifulSoup
import re
import string

# copy code from the paper
def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

# normalization copied from the paper's code
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
  
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def search(entity: str) -> str:
    """
    searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    """
    # Format the entity for URL encoding
    entity_formatted = entity.replace(" ", "+")
    url = f"https://en.wikipedia.org/w/index.php?search={entity_formatted}"
    
    # Fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check if the exact page was found or suggest similar items
    # when <div class=mw-search-result-heading> is detected, it means the entity page is not found on wikipedia
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    
    if result_divs: # this means the searched entity page is not in wikipedia, wikipedia will show a list of similar entities
        # get Similar results
        similar_titles = [div.a.get_text() for div in result_divs]
        return f"Could not find exact page for '{entity}'. Similar topics: {similar_titles[:5]}" # return the top 5 similar titles
    else:
        # the paper uses page to represent content in <p>
        # Extract xontent
        page_list = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        # TODO: Recursive search, if find any concept that needs more search then call search again
        # if any("may refer to:" in p for p in page_list):
        #     search(entity)

        # restructure & clean the page content following the paper's logic
        page = ''
        for p in page_list:
            if len(p.split(" ")) > 2:
                page += clean_str(p)
                if not p.endswith("\n"):
                    page += "\n"
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # return the first 5 sentences
        if sentences:
            return ' '.join(sentences[:5]) if len(sentences)>=5 else ' '.join(sentences)
        else:
            return "No content found on this page."
        
        # TODO: clean the paragraphs and return the searched content


def lookup(text: str, keyword: str) -> str:
    """
        returns the sentences containing keyword in the current passage.
    """
    sentences = text.split('.')
    matching_sentences = [sentence.strip() + '.' for sentence in sentences if keyword.lower() in sentence.lower()]
    if not matching_sentences:
        return "No sentences found with the keyword."
    else:
        return ' '.join(matching_sentences)  # Join all matching sentences into a single string




# set up tools for the agent
tools = [FunctionTool.from_defaults(fn=search), FunctionTool.from_defaults(fn=lookup)]


# ## Examples
# The next thing to add is examples. Few shot prompt engineering is a common practice to improve the model performance.
# 
# Let's use the paper's examples. The paper has 6 examples altogether.



examples = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: search("Colorado orogeny")
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: lookup("eastern sector")
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: search("High Plains")
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: search("High Plains (United States)")
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: finish("1,800 to 7,000 ft")""",
"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: search("Milhouse")
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: lookup("named after")
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: finish("Richard Nixon")""",
"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: search("Adam Clayton Powell")
Observation 1: Could not find ["Adam Clayton Powell"]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: search("Adam Clayton Powell (film)")
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: finish("The Saimaa Gesture")""",
"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: search("Nicholas Ray")
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: search("Elia Kazan")
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: finish("director, screenwriter, actor")""",
"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: search("Arthur's Magazine")
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: search("First for Women")
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: finish("Arthur's Magazine")""",
"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: search("Pavel Urysohn")
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: search("Leonid Levin")
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: finish("yes")"""
]




# preset up the examples as prompt_kwargs, the examples will be included in the system prompt

preset_prompt_kwargs = {"examples": examples} 


# ## Model
# 
# Next, we can choose the model to call. In this example we will use OpenAIClient ``gpt-3.5-turbo`` model. We will set the ``temperature`` at 0.0 to make the response as consistent as possible.



gpt_model_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
}


# ## Agent
# Combining the previous components, we can define the agent.



# max_steps refers to how many thought-action round we allow the model to perform
# to save resources, let's use 3 here
agent = ReActAgent(
        tools=tools, max_steps=3, model_client=OpenAIClient(),
        model_kwargs=gpt_model_kwargs, preset_prompt_kwargs=preset_prompt_kwargs
)
agent




import importlib
import adalflow
importlib.reload(adalflow)


# # 5. Q & A
# Next we can use the agent to answer our questions. Let's run 5 examples. We will use the validation data.



val_dataset = dataset["validation"]
val_dataset


# ``LightRAG`` provides a ``printc`` function. You can utilize it to show colored console output for angent.



from adalflow.utils.logger import printc

num_questions = 5
for i in range(num_questions):
    question = val_dataset[i]["question"]
    gt_answer = normalize_answer(val_dataset[i]["answer"]) # normalize the ground truth answer
    
    # get the agent's response
    pred_answer = agent(question)
    pred_answer = normalize_answer(pred_answer)
    
    printc(f"question: {question}, ground truth: {gt_answer}, pred answer: {pred_answer}", color="yellow")


# # 6. Evaluation
# 
# Now you will see that we have the ``exact correct answer`` for some questions:
# 
# question: Were Scott Derrickson and Ed Wood of the same nationality?, ground truth: ``yes`` pred answer: ``yes``
# 
# question: What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?, ground truth: ``animorphs``, pred answer: ``animorphs``
# 
# Sometimes the agent performs correctly but not in the same format with the ground truth. E.g. ground truth: ``no``, pred answer: ``no, they are not the same``. This is what we can tolerate.
# 
# But how to evaluate if the agent is doing well, or if our tools, examples, and prompt implementations work well? We need to evaluate it.
# 
# 1. Exact Match(EM)
# Exact Match is what the paper is using. Only when the normalized agent response is the same with the ground truth answer, we count it as correct. The paper's EM for react agent is around 30%(gpt-3).
# 
# 2. Fuzzy Match(FM)
# EM doesn't make much sense in question and answering. So we propose fuzzy match based on the LLMs' lengthy output nature. If the ground truth answer is included in the agent response, then we count it as correct. FM is not necessarily correct. 
# E.g. question: Harry Potter and Dumbledore, who is older? ground truth: ``dumbledore``, pred answer: ``harry potter is older than dumbledore.``
# the model mentioned the groud truth but still provide wrong answer. So FM serves as reference.
# 
# Let's use ``LightRAG eval`` module and evaluate on 10 questions and keep the model's practice to set ``max_step`` at `7`.



from adalflow.eval.answer_match_acc import AnswerMatchAcc

# set up evaluation type
EM_evaluator = AnswerMatchAcc(type="exact_match")
FM_evaluator = AnswerMatchAcc(type="fuzzy_match")

agent = ReActAgent(
        tools=tools, max_steps=7, model_client=OpenAIClient(),
        model_kwargs=gpt_model_kwargs, preset_prompt_kwargs=preset_prompt_kwargs
)

num_questions = 10
gt_answers = []
pred_answers = []
start_time = time.time()
for i in range(num_questions):
    question = val_dataset[i]["question"]
    gt_answer = normalize_answer(val_dataset[i]["answer"]) # normalize the ground truth answer
    gt_answers.append(gt_answer)
    
    # get the agent's response
    pred_answer = agent(question)
    pred_answer = normalize_answer(pred_answer)
    pred_answers.append(pred_answer)
    
    printc(f"No. {i+1}, question: {question}, ground truth: {gt_answer}, pred answer: {pred_answer}", color="yellow")

end_time = time.time()
    
em = EM_evaluator.compute(pred_answers=pred_answers, gt_answers=gt_answers)
fm = FM_evaluator.compute(pred_answers=pred_answers, gt_answers=gt_answers)
avg_time = (end_time - start_time) / num_questions

print(f"EM = {em}, FM = {fm}, average time = {avg_time}")


# The EM is 0.3 and the FM is 0.6. Each query takes 9s in average. (The performance also depends on the success of wikipedia query connection.)

# What if we use simple LLM models to answer these questions? To test on this, we just need to remove the tools. We have a built-in ``llm_tool`` and ``finish`` that automatically handles the query. ``llm_tool`` uses the same model with the agent. We can't add the examples because the examples will mislead the agent to use non-existing tools.



from adalflow.eval.answer_match_acc import AnswerMatchAcc

# set up evaluation type
EM_evaluator = AnswerMatchAcc(type="exact_match")
FM_evaluator = AnswerMatchAcc(type="fuzzy_match")

agent = ReActAgent(
        max_steps=7, model_client=OpenAIClient(),
        model_kwargs=gpt_model_kwargs
)

num_questions = 10
gt_answers = []
pred_answers = []
start_time = time.time()
for i in range(num_questions):
    question = val_dataset[i]["question"]
    gt_answer = normalize_answer(val_dataset[i]["answer"]) # normalize the ground truth answer
    gt_answers.append(gt_answer)
    
    # get the agent's response
    pred_answer = agent(question)
    pred_answer = normalize_answer(pred_answer)
    pred_answers.append(pred_answer)
    
    printc(f"No. {i+1}, question: {question}, ground truth: {gt_answer}, pred answer: {pred_answer}", color="yellow")

end_time = time.time()
    
em = EM_evaluator.compute(pred_answers=pred_answers, gt_answers=gt_answers)
fm = FM_evaluator.compute(pred_answers=pred_answers, gt_answers=gt_answers)
avg_time = (end_time - start_time) / num_questions

print(f"EM = {em}, FM = {fm}, average time = {avg_time}")


# Without the tools and examples, EM=0 and FM=0.4. We saw hallucinations and nonsense:
# 
# 2024-06-15 23:17:04 - [3230041225.py:26:<module>] - No. 1, question: Were Scott Derrickson and Ed Wood of the same nationality?, ground truth: ``yes``, pred answer: ``no scott derrickson and ed wood were not of same nationality scott derrickson is american while ed wood was also american``
# 
# 2024-06-15 23:18:16 - [3230041225.py:26:<module>] - No. 9, question: Who is older, Annie Morton or Terry Richardson?, ground truth:`` terry richardson``, pred answer: ``who is older annie morton or terry richardson``
# 
# Therefore, using ReAct agent outperforms the base LLM.
# Meanwhile, ``LightRAG ReAct agent`` shows that the performance on 10 questions(EM=0.3).

# # 7. Future Improvement



# TODO:
# 1. advanced, add history to react
# 2. add training, few shot
# 3. llm as judge
# 4. add picture
# 5. better json handling, we need to store the answer output

