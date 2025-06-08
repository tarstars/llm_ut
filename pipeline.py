# pipeline.py

import json, asyncio
from jinja2 import Template
import openai
from .models import PromptVersion, ResponseRecord

# load basket once
with open("basket.json") as f:
    BASKET = json.load(f)

# 1. generate_answers(prompt_text) → {case_id: answer_text}
# 2. evaluate_answers(answers) → {case_id: raw_eval_text}
# 3. parse_flags(raw_eval) → flags dict + final_flag
# 4. aggregate_metrics(all_flags) → metrics_summary
