from ..base import BasePrompt

class ARCPrompt(BasePrompt):
    """ARC dataset prompt for self-evaluation"""
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
You are a helpful assistant that evaluates the factual accuracy and logical consistency of explanations.
Given a question, choices and an answer, decide which of the two provided explanations is more factually grounded and logically valid.
Your output must be either JMH or BHY, where JMH corresponds to the first explanation and BHY corresponds to the second explanation.
"""
        
        self.examples = """
Question:
Which of the following statements best explains why the tilt of Earth on its axis causes summer to be warmer than winter in the Northern Hemisphere?
Choices:
A. The warm ocean currents flow from the tropics to the Northern Hemisphere in the summer.
B. The rays of the Sun strike the Northern Hemisphere more directly in the summer.
C. The greenhouse effect increases in the Northern Hemisphere in the summer.
D. The Northern Hemisphere is closer to the Sun in the summer.
Answer:
B
Explanation JMH:
The tilt of Earth on its axis causes the Northern Hemisphere to be warmer in the summer than in the winter.
This is because during summer, the Northern Hemisphere is exposed to more direct rays of the Sun than during winter.
Therefore, of the listed statements, the tilt of Earth on its axis causes summer to be warmer than winter in the Northern Hemisphere.
Explanation BHY:
In the Northern Hemisphere, the tilt of Earth's axis causes the Sun to be directly overhead in the tropics during the summer solstice.
The greenhouse effect increases in the summer, warming the Northern Hemisphere.
Your Output (JMH/BHY):
JMH

Question: 
The male insects in a population are treated to prevent sperm production. Would this reduce this insect population?
Choices: 
A. No, because the insects would still mate. 
B. No, because it would not change the offspring mutation rate. 
C. Yes, because it would sharply decrease the reproduction rate. 
D. Yes, because the males would die.
Answer: 
C
Explanation JMH:
All members of a population have the same genes. The genes of each member of the population are passed down to their offspring. If the genes of the male insects are treated to prevent sperm production, the genes of the female insects will still be passed down to each of their offspring. Thus, this treatment would not affect the reproduction rate of the insect population.
Explanation BHY:
The suppression of sperm production in males would reduce the reproduction rate of that population. The suppression of sperm production would not change the mutation rate of the population, as mutations occur in the egg and not the sperm. However, the suppression of sperm production would not cause the males to die, as the males would still be able to mate with females. Therefore, the suppression of sperm production in males would sharply reduce the reproduction rate of that population.
Your Output (JMH/BHY):
BHY

Question: 
On Earth, water can be a solid, a liquid, or a gas. Which energy source has the greatest influence on the state of matter of water?
Choices: 
A. the sun
B. the wind
C. ocean currents
D. the metal core
Answer: A
Explanation JMH:
The sun's heat source causes water to evaporate.
However, wind can cause evaporation as well as convection currents, which can transfer water vapor to different areas of the world.
Explanation BHY:
The sun's heat is the primary source of energy that causes water to exist in different states. It is the sun's heat that drives the water cycle, which is the cycle of evaporation, condensation, and precipitation that causes water to exist in solid, liquid, and gas form.
Therefore, the sun is the main energy source that shapes water's state of matter.
Your Output (JMH/BHY):
BHY

Question: 
A toothpaste commercial states that a brand of toothpaste has a higher concentration of fluoride than any other toothpaste available. The commercial is most likely inferring that the advertised toothpaste
Choices: 
A. has a pleasant flavor.
B. is recommended by dentists.
C. promotes good dental hygiene.
D. is the most expensive brand sold.
Answer: 
C
Explanation JMH:
Fluoride is a mineral found in natural water sources that promotes dental health. The advertisement is comparing the amount of fluoride in the toothpaste to the amount of fluoride in other toothpastes. This comparison implies that the brand of toothpaste advertised has the highest amount of fluoride, which is the best for dental health.
Explanation BHY:
The commercial is most likely inferring that the advertised toothpaste is the toothpaste that has the most fluoride. They state that the toothpaste has a higher concentration of fluoride than any other toothpaste available, which is an indication that it is better than other brands. The advertisers do not mention any other traits, and the only other information provided is that it has a higher concentration of fluoride. Therefore, the commercial is most likely inferring that the advertised toothpaste is the toothpaste that has the most fluoride.
Your Output (JMH/BHY):
JMH
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Question:\n{question}\n"
            "Choices:\n{choices}\n"
            "Answer:\n{gold_answer}\n"
            "Explanation JMH:\n{explanation1}\n"
            "Explanation BHY:\n{explanation2}\n"
            "Your Output (JMH/BHY):\n"
        )