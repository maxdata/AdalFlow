#!/usr/bin/env python
# coding: utf-8

# # 🤗 Welcome to AdalFlow!
# ## The PyTorch library to auto-optimize any LLM task pipelines
# 
# Thanks for trying us out, we're here to provide you with the best LLM application development experience you can dream of 😊 any questions or concerns you may have, [come talk to us on discord,](https://discord.gg/ezzszrRZvT) we're always here to help! ⭐ <i>Star us on <a href="https://github.com/SylphAI-Inc/AdalFlow">Github</a> </i> ⭐
# 
# 
# # Quick Links
# 
# Github repo: https://github.com/SylphAI-Inc/AdalFlow
# 
# Full Tutorials: https://adalflow.sylph.ai/index.html#.
# 
# Deep dive on each API: check out the [developer notes](https://adalflow.sylph.ai/tutorials/index.html).
# 
# Common use cases along with the auto-optimization:  check out [Use cases](https://adalflow.sylph.ai/use_cases/index.html).
# 
# # Outline
# This is the colab complementary to:
# * [LLM evaluation guideline](https://adalflow.sylph.ai/tutorials/evaluation.html)
# * [Source code](https://github.com/SylphAI-Inc/AdalFlow/tree/main/tutorials/evaluation)
# 
# 
# Introducing LLM evaluations with a focus on the generative tasks instead of classical Natural language understanding tasks.
# 
# * Natural language Generation(NLG) metrics
# * RAG evaluation:
#     * RAG AnswerMatch
#     * RAG Retriever Recall
# 
# 
# 
# # Installation
# 
# 1. Use `pip` to install the `adalflow` Python package. We will need `openai`, `groq`, and `faiss`(cpu version) from the extra packages.
# 
#   ```bash
#   pip install adalflow[openai,groq,faiss-cpu]
#   ```
# 2. Setup  `openai` and `groq` API key in the environment variables



# ensure version >= v0.2.1
from IPython.display import clear_output

get_ipython().system('pip install -U adalflow[openai]')

clear_output()


# ## Set Environment Variables
# 
# Run the following code and pass your api key.
# 
# Note: for normal `.py` projects, follow our [official installation guide](https://lightrag.sylph.ai/get_started/installation.html).
# 
# *Go to [OpenAI](https://platform.openai.com/docs/introduction) and [Groq](https://console.groq.com/docs/) to get API keys if you don't already have.*



import os

from getpass import getpass

# Prompt user to enter their API keys securely
openai_api_key = getpass("Please enter your OpenAI API key: ")


# Set environment variables
os.environ['OPENAI_API_KEY'] = openai_api_key

print("API keys have been set.")


# # 😇 Classical Text metrics and issues
# 
# We will use `Torchmetrics` to compute the classical text metrics like BLEU, ROUGE.
# 
# We choose a case where the ground truth(references) means the same as the generated text, but where BLEU and ROUGE are not able to capture the similarity.



get_ipython().system('pip install torchmetrics')

clear_output()




gt = "Brazil has won 5 FIFA World Cup titles"
pred = "Brazil is the five-time champion of the FIFA WorldCup."


def compute_rouge(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/rouge_score.html
    """
    from torchmetrics.text.rouge import ROUGEScore

    rouge = ROUGEScore()
    return rouge(pred, gt)


def compute_bleu(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/bleu_score.html
    """
    from torchmetrics.text.bleu import BLEUScore

    bleu = BLEUScore()
    # preds = ["the cat is on the mat"]
    # target = [["there is a cat on the mat", "a cat is on the mat"]]
    # score = bleu(preds, target)
    # print(f"score: {score}")
    # print(f"pred: {[pred]}, gt: {[[gt]]}")
    return bleu([pred], [[gt]])




compute_rouge(gt, pred)




compute_bleu(gt, pred)


# # 🤗  Embedding-based Metrics -- BERTScore
# 
# To make up for this, embedding-based  metrics or neural evaluators such as BERTScore was created.
# You can find BERTScore in both `Hugging Face Metrics <https://huggingface.co/metrics>`_ and `TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/text/bertscore.html>`_.
# BERTScore uses pre-trained contextual embeddings from BERT and matched words in generated text and reference text using cosine similarity.



def compute_bertscore(gt, pred):
    r"""
    https://lightning.ai/docs/torchmetrics/stable/text/bert_score.html
    """
    from torchmetrics.text.bert import BERTScore

    bertscore = BERTScore()
    return bertscore([pred], [gt])




compute_bertscore(gt, pred)


# # 🤗  LLM As Judge
# 
# AdalFlow provides a very customizable LLM judge, which can be used in three ways:
# 
# 1. With question, ground truth, and generated text
# 2. Without question, with ground truth, and generated text
# 3. Without question, without ground truth, with generated text
# 
# And you can customize the `judgement_query` towards your use case or even the whole llm template.
# 
# AdalFlow LLM judge returns `LLMJudgeEvalResult` which has three fields:
# 1. `avg_score`: average score of the generated text
# 2. `judgement_score_list`: list of scores for each generated text
# 3. `confidence_interval`: a tuple of the 95% confidence interval of the scores
# 
# 
# `DefaultLLMJudge` is an LLM task pipeline that takes a single question(optional), ground truth(optional), and generated text and returns the float score in range [0,1].
# 
# You can use it as an `eval_fn` for AdalFlow Trainer.
# 
# `LLMAsJudge` is an evaluator that takes a list of inputs and returns a list of `LLMJudgeEvalResult`.
# Besides of the score, it computes the confidence interval of the scores.



# without questions, and with customized judgement query

def compute_llm_as_judge_wo_questions():
    from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge
    from adalflow.components.model_client import OpenAIClient


    llm_judge = DefaultLLMJudge(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 10,
        },
        jugement_query="Does the predicted answer means the same as the ground truth answer? Say True if yes, False if no.",
    )
    llm_evaluator = LLMasJudge(llm_judge=llm_judge)
    print(llm_judge)
    eval_rslt = llm_evaluator.compute(gt_answers=[gt], pred_answers=[pred])
    print(eval_rslt)




compute_llm_as_judge_wo_questions()




# with questions and default judgement query
def compute_llm_as_judge():
    from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge
    from adalflow.components.model_client import OpenAIClient

    questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    gt_answers = ["Yes", "Yes", "No"]

    llm_judge = DefaultLLMJudge(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 10,
        },
    )
    llm_evaluator = LLMasJudge(llm_judge=llm_judge)
    print(llm_judge)
    eval_rslt = llm_evaluator.compute(
        questions=questions, gt_answers=gt_answers, pred_answers=pred_answers
    )
    print(eval_rslt)




compute_llm_as_judge()


# # 🤩 G-eval
# 
# If you have no reference text, you can also use G-eval [11]_ to evaluate the generated text on the fly.
# G-eval provided a way to evaluate:
# 
# - `relevance`: evaluates how relevant the summarized text to the source text.
# - `fluency`: the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
# - `consistency`: evaluates the collective quality of all sentences.
# - `coherence`: evaluates the the factual alignment between the summary and the summarized source.
# 
# In our library, we provides the prompt for task `Summarization` and `Chatbot` as default.
# We also map the score to the range [0, 1] for the ease of optimization.
# 
# Here is the code snippet to compute the G-eval score:



def compute_g_eval_summarization(source, summary):
    from adalflow.eval.g_eval import GEvalLLMJudge, GEvalJudgeEvaluator, NLGTask

    model_kwargs = {
        "model": "gpt-4o",
        "n": 20,
        "top_p": 1,
        "max_tokens": 5,
        "temperature": 1,
    }

    g_eval = GEvalLLMJudge(
        default_task=NLGTask.SUMMARIZATION, model_kwargs=model_kwargs
    )
    print(g_eval)
    input_template = """Source Document: {source}
    Summary: {summary}
    """

    input_str = input_template.format(
        source=source,
        summary=summary,
    )

    g_evaluator = GEvalJudgeEvaluator(llm_judge=g_eval)

    response = g_evaluator(input_strs=[input_str])
    print(f"response: {response}")




source="Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with Burnley on Sunday . 'Just been watching the game , did you miss the coach ? # RubberDub # 7minutes , ' Merson put on Twitter . Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in ( the England team ) then it opens it up to anybody . ' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England 's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake . 'It 's not as though I was watching hoping he would n't score for England , I 'm genuinely pleased for him and fair play to him \u00e2\u20ac\u201c it was a great goal , ' Merson said . 'It 's just a matter of opinion , and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson , so he should n't have been in the squad . 'When I 'm wrong , I hold my hands up . I do n't have a problem with doing that - I 'll always be the first to admit when I 'm wrong . ' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit Merson ( centre ) criticised Townsend 's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday , saying 'Not bad for a player that should be 'nowhere near the squad ' ay @ PaulMerse ? ' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor .",
summary="Paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . Andros townsend scored the tottenham midfielder in the 89th minute . Paul merson had another dig at andros townsend after his appearance . The midfielder had been brought on to the england squad last week . Click here for all the latest arsenal news news .",

compute_g_eval_summarization(source=source, summary=summary)




compute_g_eval_summarization(source=gt, summary=pred)


# # Issues and feedback
# 
# If you encounter any issues, please report them here: [GitHub Issues](https://github.com/SylphAI-Inc/LightRAG/issues).
# 
# For feedback, you can use either the [GitHub discussions](https://github.com/SylphAI-Inc/LightRAG/discussions) or [Discord](https://discord.gg/ezzszrRZvT).
