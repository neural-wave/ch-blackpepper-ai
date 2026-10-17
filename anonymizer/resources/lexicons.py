"""Small curated lexicons. Lower-case everywhere."""

# Common US first names: used to accept name candidates in lower-cased dialogue
# (e.g. "hi , andrew") where NER is unreliable.
FIRST_NAMES = set("""
aaron abigail adam adrian aiden alan albert alex alexander alexandra alexis alice alicia allison amanda amber amy
andrea andrew angela anita ann anna anne annie anthony antonio april arthur ashley audrey austin barbara benjamin
beth betty beverly bill billy bob bobby bonnie brad bradley brandon brenda brian brittany bruce bryan caleb cameron
carl carla carlos carmen carol caroline carolyn catherine charles charlie charlotte chloe chris christian christina
christine christopher cindy claire clara clarence cody colin connie craig crystal cynthia daisy dale daniel danielle
danny david dean debbie deborah debra denise dennis derek diana diane donald donna doris dorothy douglas dylan
earl edward eileen elaine eleanor elijah elizabeth ella ellen emily emma eric erica erin ethan eugene eva evelyn
frances frank fred gabriel gail gary george gerald gina gloria grace gregory hailey hannah harold harry heather
helen henry holly howard ian isaac isabella isabel jack jackie jacob jacqueline james jamie jane janet janice jason
jean jeff jeffrey jennifer jenny jeremy jerry jesse jessica jesus jill jim jimmy joan joe joel john johnny jonathan
jordan jose joseph josh joshua joyce juan judith judy julia julie justin karen katherine kathleen kathryn kathy katie
keith kelly kenneth kevin kim kimberly kyle larry laura lauren lawrence leah lee leslie lillian lily linda lisa logan
lori louis louise lucas lucy luis luke lynn madison margaret maria marie marilyn mark martha martin mary mason matthew
megan melanie melissa michael michelle mike mildred monica nancy natalie nathan nicholas nicole noah norma olivia
oscar pamela patricia patrick paul paula peggy peter philip phillip rachel ralph randy raymond rebecca regina richard
rick rita robert roberto roger ronald rose roy ruby russell ruth ryan samantha samuel sandra sara sarah scott sean
sharon shawn shirley sophia stephanie stephen steve steven susan tammy teresa terry theresa thomas tiffany timothy
tina todd tom tony tracy tyler valerie vanessa victor victoria vincent virginia walter wanda wayne wendy william
willie zachary sophie mia ava harper evelyn abby zoe ruby maya nora stella hazel violet aurora penelope layla riley
sebastian jayden carter wyatt julian levi isaiah lincoln jaxon asher leo ezra hudson miles everett easton
""".split())

# Words that look like names after a greeting/title but are not ("hi there", "thanks doc").
NOT_NAMES = set("""
there doc doctor sir maam ma'am everyone everybody guys guy man buddy again so how okay ok yes no well um uh hi hey
hello good morning afternoon evening thanks thank you nice great all right alright just come in welcome bye see
sweetie honey dear friend folks team nurse patient mom dad son daughter wife husband today now then here
dragon dax nuance dr mr mrs ms miss mister mm hmm mhm mm-hmm uh-huh uh-uh huh ah oh
""".split())

# Ambiguous names that are also common words: only accepted with strong evidence (title / NER / capitalised).
AMBIGUOUS_NAMES = set("""
will may june april august mark bill grace hope joy faith rose lily ruby amber crystal dawn summer autumn
art sunny chase drew frank gene jack max miles pat ray rich rocky sky stone win carol
""".split())

# Eponyms that name diseases, signs, tests or structures. A name token in this list is medical
# when it sits next to one of EPONYM_CONTEXT words (or inside a medical entity).
EPONYMS = set("""
alzheimer alzheimers parkinson parkinsons crohn crohns hashimoto hashimotos graves bell bells cushing cushings
addison addisons hodgkin hodgkins raynaud raynauds sjogren sjogrens tourette tourettes huntington huntingtons down
asperger aspergers baker bakers barrett barretts kawasaki lyme meniere menieres osgood schlatter dupuytren dupuytrens
quervain guillain barre marfan wilson wilsons paget pagets peyronie reiter hawkins kennedy neer neers babinski phalen
phalens tinel tinels mcmurray lachman homan homans murphy murphys apley finkelstein spurling romberg valsalva epley
heberden bouchard colles smith achilles morton mortons bartholin dix hallpike trendelenburg thompson allen allens
lhermitte durkan hoffman hoffmann kernig brudzinski cullen turner kaposi wernicke korsakoff bowen sever haglund
""".split())
EPONYM_CONTEXT = set("""
disease diseases syndrome sign signs test tests maneuver manoeuvre fracture cyst palsy esophagus neuroma tendon
tendonitis tendinitis contracture phenomenon reflex node nodes lymphoma thyroiditis's 's dementia tremor tremors
procedure position classification scale score criteria ulcer gland duct
""".split())

# Occupations: quasi-identifiers, generalized to a sector.
OCCUPATIONS = {
    "teacher": "education", "professor": "education", "principal": "education", "student": "education",
    "nurse": "healthcare", "physician": "healthcare", "paramedic": "healthcare", "pharmacist": "healthcare",
    "dentist": "healthcare", "surgeon": "healthcare", "caregiver": "healthcare",
    "firefighter": "public service", "police officer": "public service", "cop": "public service",
    "soldier": "public service", "veteran": "public service", "mail carrier": "public service",
    "lawyer": "professional services", "attorney": "professional services", "accountant": "professional services",
    "engineer": "technical", "programmer": "technical", "software developer": "technical", "electrician": "trades",
    "plumber": "trades", "carpenter": "trades", "mechanic": "trades", "construction worker": "trades",
    "contractor": "trades", "roofer": "trades", "welder": "trades", "landscaper": "trades", "painter": "trades",
    "farmer": "agriculture", "rancher": "agriculture", "truck driver": "transport", "pilot": "transport",
    "bus driver": "transport", "delivery driver": "transport", "flight attendant": "transport",
    "chef": "hospitality", "cook": "hospitality", "waiter": "hospitality", "waitress": "hospitality",
    "bartender": "hospitality", "barista": "hospitality", "cashier": "retail", "salesman": "retail",
    "realtor": "sales", "banker": "finance", "hairdresser": "personal services", "barber": "personal services",
    "musician": "arts", "artist": "arts", "actor": "arts", "writer": "arts", "photographer": "arts",
    "coach": "sports", "athlete": "sports", "personal trainer": "sports", "pastor": "religious", "priest": "religious",
}

# Units marking a measurement value that can be perturbed (not doses).
MEASUREMENT_UNITS = r"(?:%|percent|mmhg|bpm|beats per minute|kg|kilograms|lbs|pounds|mg/dl|milligrams per deciliter|degrees|°f|°c|cm|centimeters)"
