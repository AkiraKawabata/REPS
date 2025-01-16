from ..base import BasePrompt

class STRATEGY_QAPrompt(BasePrompt):
    """ARC dataset prompt for self-evaluation"""
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
You are a helpful assistant that evaluates the factual accuracy and logical consistency of explanations.
Given a question and an answer, decide which of the two provided explanations is more factually grounded and logically valid.
Your output must be either JMH or BHY, where JMH corresponds to the first explanation and BHY corresponds to the second explanation.
"""
        
        self.examples = """
Question:
Is the current Chief Justice of the United States forbidden from buying alcohol?
Answer:
no
Explanation JMH:
The current Chief Justice of the United States is John Roberts.
John Roberts is a citizen of the United States.
Citizens of the United States are allowed to purchase alcohol.
Thus, the current Chief Justice of the United States is allowed to buy alcohol.
Explanation BHY:
John Roberts is the current Chief Justice of the United States.
Thus, Roberts is forbidden from buying alcohol.
Your Output (JMH/BHY):
JMH

Question:
Was Noah concerned with buoyancy?
Answer:
yes
Explanation JMH:
Noah’s flood lasted for 40 days.
The buoyancy of water is roughly 1.0g/cm3.
Thus, Noah was concerned with buoyancy.
Explanation BHY:
Noah was concerned with the Ark.
The Ark was buoyant.
Thus, Noah was concerned with buoyancy.
Your Output (JMH/BHY):
BHY

Question:
Did Rumi spend his time in a state of euphoria?
Answer:
yes
Explanation JMH:
Rumi is supposed to have had a mystical experience where he was in a state of euphoria.
A mystical experience is supposed to be a spiritual experience.
Thus, Rumi spent his time in a state of euphoria.
Explanation BHY:
Rumi was a Sufi mystic.
Sufi mystics have a state known as "majzoob" or "majzub".
This is a state of euphoria.
Your Output (JMH/BHY):
BHY

Question:
Does a sea otter eat spiders?
Answer:
no
Explanation JMH:
A sea otter eats clams, snails, and sea urchins.
Spiders are not among these.
Explanation BHY:
Sea otters sometimes eat spiders.
Thus, sea otters eat spiders.
Your Output (JMH/BHY):
JMH
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Question:\n{question}\n"
            "Answer:\n{gold_answer}\n"
            "Explanation JMH:\n{explanation1}\n"
            "Explanation BHY:\n{explanation2}\n"
            "Your Output (JMH/BHY):\n"
        )