from ..base import BasePrompt

class DROPPrompt(BasePrompt):
    
    def __init__(self):
        super().__init__()
        
        self.instruction = """
Analyze the given passage and question carefully. Provide a step-by-step explanation of your reasoning process to arrive at the correct answer. After explaining your logic, state the final answer.
        """
        
        self.examples = """
Passage:
Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.
Question:
How many games had the Bills won before this game?
Explanation:
The passage mentions that the Bills were trying to snap a two-game skid. 
This implies that they had lost their previous two games before this game against the New England Patriots.
Therefore, they had not won any games before this one.
Answer:
0

Passage:
The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.
Question:
How many years before Navarrase used the captivity of John II?
Explanation:
The passage states that John II was captured and that the Navarrese used his captivity to attempt seizing power in 1363. 
The passage also mentions that the French crown had been at odds with Navarre since 1354.
The difference between 1363 (when the Navarrese attempted to seize power) and 1354 (when the conflict with Navarre began) is 9 years.
Answer:
9

Passage:
As of the census of 2000, there were 218,590 people, 79,667 households, and 60,387 families residing in the county. The population density was 496 people per square mile (192/km²). There were 83,146 housing units at an average density of 189 per square mile (73/km²). The racial makeup of the county was 86.77% Race (United States Census), 9.27% Race (United States Census), 0.23% Race (United States Census), 1.52% Race (United States Census), 0.06% Race (United States Census), 0.69% from Race (United States Census), and 1.47% from two or more races. 1.91% of the population were Race (United States Census) or Race (United States Census) of any race. 22.5% were of German people, 13.1% Irish people, 9.8% Italian people, 9.2% English, 8.1% "American" and 6.0% Polish ancestry.
Question:
Which group from the census is smaller: German or Irish?
Explanation:
The passage states that 22.5% of the population were of German ancestry and 13.1% were of Irish ancestry.
Therefore, the group of Irish people is smaller than the group of German people.
Answer:
Irish

Passage:
Hoping to rebound from their tough overtime road loss to the Raiders, the Jets went home for a Week 8 duel with the Kansas City Chiefs. In the first quarter, New York took flight as QB Brett Favre completed an 18-yard TD pass to RB Leon Washington. In the second quarter, the Chiefs tied the game as QB Tyler Thigpen completed a 19-yard TD pass to TE Tony Gonzalez. The Jets would answer with Washington getting a 60-yard TD run. Kansas City closed out the half as Thigpen completed an 11-yard TD pass to WR Mark Bradley. In the third quarter, the Chiefs took the lead as kicker Connor Barth nailed a 30-yard field goal, yet New York replied with RB Thomas Jones getting a 1-yard TD run. In the fourth quarter, Kansas City got the lead again as CB Brandon Flowers returned an interception 91 yards for a touchdown. Fortunately, the Jets pulled out the win with Favre completing the game-winning 15-yard TD pass to WR Laveranues Coles. During halftime, the Jets celebrated the 40th anniversary of their Super Bowl III championship team.
Question:
Who threw the second longest touchdown pass?
Explanation:
The longest TD pass was also thrown by Thigpen, a 19-yard pass to Tony Gonzalez.
Brett Favre's two TD passes were 18 yards to Leon Washington and 15 yards to Laveranues Coles, both shorter than Thigpen's 11-yard pass to Bradley.
Therefore, Thigpen's 11-yard pass to Bradley was the second longest touchdown pass in the game.
Explanation:
By listing the touchdown passes in order of their length:
Tyler Thigpen’s 19-yard pass is the longest.
Brett Favre’s 18-yard pass is the second longest.
Brett Favre’s 15-yard pass is the third longest.
Tyler Thigpen’s 11-yard pass is the shortest.
Thus, Brett Favre threw the second longest touchdown pass in the game, an 18-yard completion to Leon Washington. 
Answer:
Brett Favre
"""
        
        self.template = (
            f"{self.instruction}\n\n"
            f"{self.examples}\n\n"
            "Passage:\n{passage}\n"
            "Question:\n{question}\n"
        )