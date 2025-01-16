from ..base import BasePrompt

class STRATEGY_QAPrompt(BasePrompt):
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
Solve given question using step-by-step inferences. Output the final answer as either Yes or No.
        """
        
        self.examples = """
Question:
Do hamsters provide food for any animals?
Explanation:
Hamsters are prey animals.
Prey are food for predators.
Thus, hamsters provide food for some animals.
Answer:
Yes

Question:
Could Brooke Shields succeed at University of Pennsylvania?
Explanation:
Brooke Shields went to Princeton University. 
Princeton University is about as academically rigorous as the University of Pennsylvania. 
Thus, Brooke Shields could also succeed at the University of Pennsylvania. 
Answer:
Yes

Question:
Hydrogen's atomic number squared exceeds number of Spice Girls?
Explanation:
Hydrogen has an atomic number of 1. 1 squared is 1. 
There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.
Answer:
No

Question:
Is it common to see frost during some college commencements?
Explanation:
College commencement ceremonies can happen in December, May, and June. 
December is in the winter, so there can be frost. 
Thus, there could be frost at some commencements.
Answer:
Yes

Question:
Could a llama birth twice during War in Vietnam (1945-46)?
Explanation:
The War in Vietnam was 6 months. 
The gestation period for a llama is 11 months, which is more than 6 months. 
Thus, a llama could not give birth twice during the War in Vietnam.
Answer:
No

Question:
Would a pear sink in water?
Explanation:
The density of a pear is about 0.6g/cm3, which is less than water. 
Objects less dense than water float. 
Thus, a pear would float.
Answer:
No
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Question:\n{question}\n"
        )