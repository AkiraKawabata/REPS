from ..base import BasePrompt

class DROPPrompt(BasePrompt):
    """ARC dataset prompt for self-evaluation"""
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
You are a helpful assistant that evaluates the factual accuracy and logical consistency of explanations.
Given a passage, question and an answer, decide which of the two provided explanations is more factually grounded and logically valid.
Your output must be either JMH or BHY, where JMH corresponds to the first explanation and BHY corresponds to the second explanation.
"""
        
        self.examples = """
Passage:
Wholesale and retail trade is the largest sector in the citys economy, contributing 24.0 percent of Bangkoks gross provincial product. It is followed by manufacturing (14.3%); real estate, renting and business activities (12.4%); transport and communications (11.6%); and financial intermediation (11.1%). Bangkok alone accounts for 48.4 percent of Thailands service sector, which in turn constitutes 49.0 percent of GDP. When the Bangkok Metropolitan Region is considered, manufacturing is the most significant contributor at 28.2 percent of the gross regional product, reflecting the density of industry in the Bangkoks neighbouring provinces. Automotive industry in Thailand based around Greater Bangkok is the largest production hub in Southeast Asia. Tourism is also a significant contributor to Bangkoks economy, generating ฿427.5bn ($13.38bn) in revenue in 2010.
Question: 
How many in percent of Bangkok's economy isn't wholesale and retail trade?
Answer:
['76']
Explanation JMH:
The passage states that wholesale and retail trade is the biggest sector in Bangkok's economy (24.0%).
Since 24.0% is the biggest, the smallest sector of Bangkok's economy must be 100% minus 24.0%, or 76%.
Explanation BHY:
Bangkoks economy is 24.0% wholesale and retail trade.
This is the percentage of Bangkoks economy that is wholesale and retail trade.
100% - 24.0% = 76.0%
76.0% of Bangkoks economy is not wholesale and retail trade.
Your Output (JMH/BHY):
BHY

Passage:
In the next 4 years, the Saudi ruler was preoccupied with consolidation of his domain, undertaking several campaigns in new regions of Arabia, while keeping the Hejazi frontier quiet. Jabal Shammar was annexed in 1920-21, while Kuwait was defeated in 1922, defining the border with Iraq and Transjordan through the Uqair protocol of 1922, while simultaneously conquering Asir in south Arabia. By early 1923, Ibn Saud decided to take over Hejaz, but was unsure over British position. The worsening relations between England and Hashemite rulers and the proclamation of Sharif Husayn as Caliph, finally made Ibn Saud to undertake the campaign, entusiasthically supported by the religiously insigted Ikhwan, who had hoped to take over the holy sites of Islam. The preliminary attack on Taif came in September 1924, beginning the Saudi conquest, which would be complete on December 1925.
Question: 
How many years do these events span for?
Answer:
['5']
Explanation JMH:
The passage starts in 1920, when the Saudi ruler was preoccupied with consolidation of his domain, undertaking several campaigns in new regions of Arabia, while keeping the Hejazi frontier quiet.
It continues with Jabal Shammar being annexed in 1920-21, while Kuwait was defeated in 1922, defining the border with Iraq and Transjordan through the Uqair protocol of 1922.
The passage concludes with the Saudi conquest being complete on December 1925.
Therefore, the events span 5 years.
Explanation BHY:
The passage states that the Saudi ruler was preoccupied with consolidation of his domain during the next 4 years.
Therefore, the next four years are from 1920 to 1924.
The passage also states that Ibn Saud decided to take over Hejaz in early 1923, which occurred before the 4 years ended.
Your Output (JMH/BHY):
JMH

Passage:
Hoping to rebound from their divisional road loss to the Packers, the Vikings' Week 11 opponent was the Oakland Raiders, who had former Vikings quarterback Daunte Culpepper under center. After a 79-yard pass from wide receiver Sidney Rice to Visanthe Shiancoe on the first play from scrimmage, the Vikings scored on the very next play on a 10-yard run from Chester Taylor. This was followed by a safety when Culpepper was penalized for intentional grounding in his own endzone. Two plays after the ensuing free kick, the Vikings fumbled the ball just inside Oakland territory, allowing the Raiders to set up a 42-yard field goal for Sebastian Janikowski. On the next drive, the Vikings restored their nine-point lead as kicker Ryan Longwell hit a 30-yard field goal on the first play of the second quarter. A short Oakland drive culminating in 10-yard touchdown pass from Culpepper to tight end John Madsen, followed by another Janikowski field goal, saw the Raiders take the lead for the first time. Four plays later, Minnesota regained a six-point lead on a 38-yard touchdown run from Taylor, but field goals of 42 and 49 yards from Janikowski meant the first half ended with the scores level at 19-19. The Vikings recorded the only score of the third quarter on a 38-yard field goal from Longwell, though they did finish the period on the Raiders' 6-yard line, allowing Taylor to run in his third touchdown on the opening play of the fourth quarter, the first time in his career that he scored three touchdowns in one game. A 52-yard field goal from Janikowski narrowed the margin to 7 points with less than three-and-a-half minutes to play, and after forcing the Vikings to punt just inside the two-minute warning, they had one last chance to level the scores. On the first play of the drive, Culpepper threw the ball in the direction of Justin Fargas, who tipped it up, allowing Chad Greenway to come up with an interception; he went to ground with the ball, but inexplicably got up and attempted to advance it, which allowed left tackle Barry Sims to force a fumble, recovered by right guard Paul McQuistan. That enabled the Raiders to extend their drive, but although they managed to get into Vikings territory, a false start penalty meant Culpepper had to attempt a Hail Mary pass on the final play, but it came up short, giving the Vikings a 29-22 win.
Question: 
How many points were scored in the third quarter?
Answer:
['3']
Explanation JMH:
The passage states that the Vikings recorded the only score of the third quarter on a 38-yard field goal from Longwell. 
Therefore, the Vikings scored 3 points in the third quarter.
Explanation BHY:
Jerry Rice, the Hall of Fame wide receiver, played for the Oakland Raiders in this game.
The passage mentions Rice's touchdown. 
Because he was on the Raiders, the Vikings' opponent, he scored a touchdown against the Vikings. 
Therefore, three touchdowns were scored in the third quarter.
Your Output (JMH/BHY):
JMH

Passage:
San Diego (4-7) led 13-3 with 7:51 remaining in the game before losing 16-13 in overtime to the Baltimore Ravens (9-2). The Ravens faced fourth-and-29 at their own 37 with 1:37 left when running back Ray Rice caught a pass one yard past the line of scrimmage. He ran to the 50-yard line, where he made three Chargers miss with a 90-degree cut to his left. Rice gained the 29 yards needed for a first down after escaping safety Eric Weddle, who received a concussion on the play as he was blocked by wide receiver Anquan Boldin. The Ravens kicked a 38-yard field goal to tie the game at the end of regulation, and made another 38-yarder to win with 1:07 left in overtime. Rivers threw a 21-yard touchdown to Floyd and Novak kicked two field goals for the Chargers' scores. Rivers was sacked six times, but did not have a turnover for only the third time in 11 games; he entered the contest with 14 interceptions and four lost fumbles. The Ravens' fourth-and-29 was the longest fourth-down conversion in the NFL since 2001. It was the third time the Chargers blew a double-digit lead in the second half, including the earlier back-to-back losses to the Saints and Broncos. After being 8-0 under Turner in November from 2009-2010, San Diego fell to 1-7, the second-worst November record in the league since 2011.
Question:
Which teams beat San Diego back to back earlier in this season?
Answer:
['the saints', 'broncos']
Explanation JMH:
The Ravens beat San Diego back to back previously in this season.
The Broncos beat them back to back earlier in this season.
The Saints beat them back to back in the previous season.
Explanation BHY:
The highlighted text in the passage says "it was the third time the Chargers blew a double-digit lead in the second half, including the earlier back-to-back losses to the Saints and Broncos."
The passage does not mention any other team that beat San Diego back-to-back earlier in this season. 
The passage only mentions two teams that beat San Diego before the game:
Broncos
Saints
Your Output (JMH/BHY):
BHY
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Passage:\n{passage}\n"
            "Question:\n{question}\n"
            "Answer:\n{gold_answer}\n"
            "Explanation JMH:\n{explanation1}\n"
            "Explanation BHY:\n{explanation2}\n"
            "Your Output (JMH/BHY):\n"
        )