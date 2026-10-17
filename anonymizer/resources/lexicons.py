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
    "minister": "religious", "paralegal": "professional services", "secretary": "office", "receptionist": "office",
    "office worker": "office", "office manager": "office", "administrative assistant": "office", "clerk": "office",
    "manager": "office", "consultant": "professional services", "analyst": "professional services",
    "architect": "technical", "designer": "technical", "developer": "technical", "scientist": "technical",
    "researcher": "technical", "technician": "technical", "machinist": "trades", "lineman": "trades",
    "nanny": "personal services", "babysitter": "personal services", "housekeeper": "personal services",
    "janitor": "facilities", "custodian": "facilities", "security guard": "public service", "sheriff": "public service",
    "deputy": "public service", "judge": "public service", "social worker": "public service",
    "postal worker": "public service", "librarian": "education", "teacher's aide": "education",
    "warehouse worker": "logistics", "factory worker": "manufacturing", "server": "hospitality",
    "dishwasher": "hospitality", "uber driver": "transport", "driver": "transport", "miner": "industry",
    "fisherman": "agriculture", "logger": "agriculture", "gardener": "trades", "groundskeeper": "trades",
    "journalist": "media", "reporter": "media", "editor": "media", "homemaker": "home",
    "stay-at-home mom": "home", "stay at home mom": "home", "marine": "public service", "in the army": "public service",
    "in the navy": "public service", "in the military": "public service", "in the air force": "public service",
}

# Units marking a measurement value that can be perturbed (not doses).
MEASUREMENT_UNITS = r"(?:%|percent|mmhg|bpm|beats per minute|kg|kilograms|lbs|pounds|mg/dl|milligrams per deciliter|degrees|°f|°c|cm|centimeters)"

# Untyped biomedical NER (en_core_sci_md) also tags conversation, time and admin words; these are never PHI or
# code evidence, and removing them would only damage the text.
GENERIC_TERMS = set("""
okay ok yeah yes no nope sure right good great fine guess mean correct excellent pleasant talk talking news
patient patient's patients doctor nurse people family myself male female man woman everything everyone something
day days week weeks month months year years hour hours minutes time morning evening night weekend weekends
summer winter spring fall today daily everyday once annual long past current
symptoms symptom problem problems issue issues questions question concerns results result reports report states
plan review reviewed reviewing assessment evaluation evaluate assess impression findings evidence diagnosis
exam examination history notes record information discussion visit appointment follow-up schedule scheduled
office clinic home house outside inside room
medical medicine medication medications treatment management care regimen prescription prescribed refill dose
changes change increase increased decrease improve improved improvement improving progress worse worsen worsening
stable normal abnormal positive negative significant unremarkable unchanged intact elevated high higher low
mild moderate severe active healthy sick consistent associated level amount size rate area side left bilateral
bilaterally sounds sound dragon
""".split())
GENERIC_PHRASES = {
    "physical exam", "physical examination", "medical history", "past medical history", "review of systems",
    "patient reports", "patient states", "patient education", "vital signs", "vitals", "blood pressure",
    "blood pressures", "heart rate", "respiratory rate", "room air", "follow up", "side effects", "normal limits",
    "treatment options", "medical treatment", "primary care physician", "pcp", "sounds good", "okay yeah",
}
# Head words that give an untyped entity a kind, used for its replacement tag.
ENTITY_KINDS = (
    ("test", r"panel|profile|count|levels?|a1c|cbc|bmp|cmp|labs?|blood ?work|urinalysis|culture|swab|biopsy|tests?"
             r"|function|screening|ua\b"),
    ("procedure", r"\w+(?:gram|graphy|scopy|ectomy|otomy|ostomy|plasty|pexy)s?\b|x-?rays?|mri|ct\b|cat scan|scans?"
                  r"|ultrasound|ekg|ecg|echo|egd|surgery|repair|release|fusion|replacement|transplant|injection"
                  r"|therapy|vaccines?|stents?|implant|pacemaker|defibrillator|dialysis|exam\b"),
    ("exam finding", r"murmurs?|bruits?|rales|rhonchi|wheez\w*|gallops?|rubs?|tenderness|palpation|auscultation"
                     r"|range of motion|reflexes"),
)

# ---------------------------------------------------------------- quasi-identifiers (phi_context.py)
# Holidays and named events date an episode ("since labor day").
HOLIDAYS = """
christmas|christmas eve|christmas day|thanksgiving|easter|halloween|labor day|memorial day|fourth of july|4th of july|
independence day|new year's|new years|new year's eve|new year's day|valentine's day|mother's day|father's day|
super bowl|hanukkah|passover|ramadan|the holidays|holiday season|holiday weekend|over the holiday|spring break|
the pandemic|covid-19 pandemic|covid pandemic|black friday|st. patrick's day|saint patrick's day|the masters|
thanksgiving break|christmas break|winter break|summer break
""".replace("\n", "").split("|")

# Sports and hobbies that single a person out; generalized unless they are the mechanism of a coded injury.
ACTIVITIES = """
soccer|football|basketball|baseball|softball|volleyball|tennis|pickleball|racquetball|squash|badminton|golf|golfing|
hockey|lacrosse|rugby|cricket|bowling|boxing|kickboxing|wrestling|karate|judo|jiu-jitsu|jiu jitsu|martial arts|
gymnastics|cheerleading|track and field|cross country|marathon|marathons|half marathon|triathlon|triathlons|
ironman|crossfit|weightlifting|powerlifting|bodybuilding|pilates|yoga|barre|barre classes|zumba|spin class|
spin classes|spinning class|cycling|mountain biking|mountain bike|biking|bike rides|bike ride|skiing|ski|skied|
snowboarding|snowboard|ice skating|skating|rollerblading|skateboarding|surfing|paddle boarding|paddleboarding|
kayaking|canoeing|rowing|sailing|swimming|scuba diving|scuba|rock climbing|bouldering|hiking|hike|hikes|
hiked|backpacking|camping|fishing|hunting|hunter|horseback riding|riding horses|dancing|ballet|salsa dancing|
gardening|knitting|crocheting|sewing|quilting|pottery|woodworking|baking|photography|birdwatching|bird watching|
chess|video games|gaming|guitar|piano|drums|choir|marching band|jogging|jog|running club|trail running|
trampoline|trampoline park|water skiing|jet ski|jet skiing|snowmobiling|motocross|dirt biking|archery|fencing|
equestrian|figure skating|ice hockey|field hockey|flag football|tee ball|t-ball|little league|travel ball|
travel team|club team|varsity|jv|intramural|beach volleyball|frisbee|ultimate frisbee|disc golf
""".replace("\n", "").split("|")
# Everyday movements: hobby only in a hobby frame (see phi_context.ACTIVITY_FRAME).
ACTIVITY_GENERIC = """
walk|walking|walks|run|running|runs|swim|swims|exercise|exercising|work out|working out|workout|workouts|
stretching|lift|lifting|lift weights|lifting weights|go to the gym|the gym|gym|treadmill|elliptical|bike|biking|
ride my bike|ride the bike|ride bikes|garden|paint|painting|cook|cooking|read|reading|travel|traveling|travelling|
dance|sing|singing|play music|play golf|play tennis|play ball|play sports|sports|meditation|meditate
""".replace("\n", "").split("|")

# Stores, chains, brands, insurers and employers that locate or profile a person.
BRANDS = """
starbucks|venti|home depot|lowe's|lowes|walgreens|cvs|rite aid|walmart|costco|kroger|safeway|publix|whole foods|
trader joe's|metro market|pick 'n save|pick n save|pick'n save|target store|mcdonald's|mcdonalds|big mac|burger king|
wendy's|taco bell|chick-fil-a|chipotle|dunkin|dunkin donuts|krispy kreme|subway sandwich|red bull|monster energy|
gatorade|mountain dew|nike|asics|new balance|hoka|hoka's|hokas|adidas|under armour|brooks|saucony|rei|
iphone|apple watch|fitbit|garmin|peloton|amazon|netflix|disney|disney world|disneyland|kaiser|blue cross|
blue cross blue shield|aetna|cigna|humana|united healthcare|enterprise rental|enterprise|uber|lyft|fedex|ups|usps|
lion's club|lions club|rotary club|ymca|planet fitness|la fitness|orangetheory|weight watchers|ww|noom|
e.l.f.|elf facial cleanser
""".replace("\n", "").split("|")
# Professional team nicknames that are unambiguous on their own.
SPORTS_TEAMS = """
celtics|lakers|knicks|clippers|mavericks|mavs|nuggets|timberwolves|pelicans|grizzlies|raptors|76ers|sixers|
cavaliers|cavs|pistons|pacers|blazers|trail blazers|cowboys|patriots|steelers|packers|seahawks|49ers|niners|
broncos|ravens|bengals|texans|jaguars|redskins|buccaneers|bucs|falcons|vikings|yankees|red sox|mets|dodgers|
cubs|white sox|astros|braves|phillies|orioles|blue jays|mariners|padres|rockies|diamondbacks|marlins|bruins|
canadiens|maple leafs|red wings|blackhawks|flyers|islanders|sabres|oilers|canucks|golden knights|kraken|
coyotes|hockey town|the meds
""".replace("\n", "").split("|")
# Team names that are also ordinary words: only after "the" and with sports words nearby.
SPORTS_TEAMS_AMBIGUOUS = ("heat|jazz|magic|nets|bulls|hawks|suns|sun|wild|rangers|stars|sky|storm|fever|sparks|"
                          "jets|bills|rays|reds|giants|twins|tigers|cardinals|angels|saints|chiefs|titans|rams|bears|lions|eagles|browns|colts|dolphins|panthers|blues|ducks|sharks|flames|devils|lightning|hurricanes|thunder|rockets|warriors|pirates|royals|brewers|guardians|senators|capitals|predators|avalanche|penguins|athletics|nationals|commanders|raiders|chargers|spurs|bucks|wizards|hornets|kings|indians").split("|")
