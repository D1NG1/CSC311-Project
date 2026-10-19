"""Column names, option lists, and regex patterns used by preprocess.py."""
import re

# Survey question -> short column name
COLUMNS = {
    "id": "id",  # student id, not a feature
    "How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.": "ease",  # rating 0-10
    "Success in this sport depends heavily on strategy and planning.": "strategy",  # Likert
    "Players must be comfortable with a high risk of injury.": "injury",  # Likert
    "Individual brilliance can outweigh teamwork in this sport.": "individual",  # Likert
    "Where might you play this sport?": "place",  # multi-select
    "During which seasons would you be most likely to play this sport recreationally in Toronto?": "season",  # multi-select
    "What types of music do you most associate with this sport?": "music",  # multi-select
    "What foods and drinks do you associate with watching this sport in the place or culture you know best?": "food",  # free text
    "What colour do you associate with this sport?": "colour",  # free text
    "Which song do you associate with this sport?": "song",  # free text
    "How do you feel when you watch this sport?": "feeling",  # free text
    "How much would you be willing to pay for a decent seat at a professional competition of this sport?": "price",  # free text (number)
    "label": "label",  # target
}

# Columns grouped by type
RATING_COL = "ease"
LIKERT_COLS = ["strategy", "injury", "individual"]
MULTI_COLS = ["place", "season", "music"]
TEXT_COLS = ["food", "colour", "song", "feeling"]
PRICE_COL = "price"

# Free-text answers that mean "no answer
MISSING_PATTERN = re.compile(
    r"^\W*(n/?a|none|nothing|no|null|idk|i don'?t know|don'?t know|no idea|"
    r"not sure|unsure|no song|no particular \w*|-+|\?+)\W*$",
    re.IGNORECASE,
)

ZERO_PRICE_PATTERN = re.compile(
    r"\b(none|nothing|zero|free|unwilling|not? (really )?willing|wouldn'?t|won'?t|"
    r"would not|not pay|no money|pay me)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(k\b)?", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"\d\s*(-|to|~)\s*\$?\s*\d", re.IGNORECASE)

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

# Fixed options of each multi-select question
MULTI_OPTIONS = {
    "place": [
        "Indoor facility",
        "Outdoor public space",
        "School or university facility",
        "Community recreation facility",
        "Private sports facility",
        "At home or nearby",
        "Others"
        ],
    "season": [
        "Winter",
        "Spring",
        "Summer",
        "Fall"
        ],
    "music": [
        "Pop",
        "Rock",
        "Hip-hop or rap",
        "Electronic or dance",
        "Latin",
        "Country",
        "Folk or traditional",
        "Classical or jazz",
        "No particular type",
        "Other"
        ],
}
