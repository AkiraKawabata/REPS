from ..base import BasePrompt

class ARCPrompt(BasePrompt):
    """ARC dataset prompt for CoT generation"""
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
Solve the given question using step-by-step inferences. After the explanation, output the correct answer from the choices labeled A through D.
        """
        
        self.examples = """
Question:
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
Choices:
A. dry palms
B. wet palms
C. palms covered with oil
D. palms covered with lotion
Explanation:
Rubbing hands together generates heat from friction.
Friction is greater between surfaces with more grip.
Wet, oily, or lotioned palms would be more slippery than dry palms.
Slippery surfaces have less friction and thus generate less heat.
Dry palms have the most friction and would generate the most heat.
Answer: 
A

Question:
Which of the following statements best explains why magnets usually stick to a refrigerator door?
Choices:
A. The refrigerator door is smooth.
B. The refrigerator door contains iron.
C. The refrigerator door is a good conductor.
D. The refrigerator door has electric wires in it.
Explanation:
Magnets stick to materials that have magnetic properties.
Iron is a common material with magnetic properties.
Thus, magnets usually stick to a refrigerator door because it contains iron.
Answer:
B

Question:
As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?
Choices:
A. 31 pounds and 14 kilograms 
B. 31 pounds and 84 kilograms 
C. 186 pounds and 14 kilograms 
D. 186 pounds and 84 kilograms
Explanation:
The astronaut's mass is 84 kilograms on the Moon and Earth, as mass does not change with location.
The Moon's gravity is about 1/6 that of Earth's. So if the astronaut weighs 31 pounds on the Moon, they would weigh about 6 times that, or 186 pounds, on Earth.
Therefore, the astronaut's approximate weight and mass on Earth are 186 pounds and 84 kilograms respectively.
Answer:
D

Question:
Which of the following is a trait that a dog does NOT inherit from its parents?
Choices:
A. the length of its fur
B. the shape of its nose 
C. the size of its appetite
D. the color of its fur
Explanation:
Physical traits like fur length, nose shape, and fur color are genetically determined and inherited from a dog's parents. However, appetite size is influenced by non-genetic factors such as diet, exercise, and health.
Therefore, of the listed traits, the size of a dog's appetite is not directly inherited from its parents.
Answer: C
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Question:\n{question}\n"
            "Choices:\n{choices}\n"
        )