from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


def hS(n, h_n=3):
  return "{} {}:".format("#" * h_n, n)


dict_h_base_h = {
    "Instruction": hS("Instruction"),
    "User": hS("User"),
    "Response": hS("Response"),
    "History": hS("Summary of Conversation History"),
}
print(dict_h_base_h)


def escapeBraces(s):
  return s.replace("{", "{{").replace("}", "}}")


class ScoreOutput(BaseModel):
  creativity: int = Field(description="creativity score", ge=1, le=10)
  creativity_justification: str = Field(description="justification for creativity score",
                                        max_length=1000)
  feasibility: int = Field(description="feasibility score", ge=1, le=10)
  feasibility_justification: str = Field(description="justification for feasibility score",
                                         max_length=1000)
  impact: int = Field(description="impact score", ge=1, le=10)
  impact_justification: str = Field(description="justification for impact score", max_length=1000)
  clarity: int = Field(description="clarity score", ge=1, le=10)
  clarity_justification: str = Field(description="justification for clarity score", max_length=1000)


score_parser = PydanticOutputParser(pydantic_object=ScoreOutput)

score_format_instruction = (
    """The output should be formatted as a JSON instance that conforms to the JSON schema below:
```
{"creativity": {"title": "Creativity", "description": "creativity score", "minimum": 1, "maximum": 10, "type": "integer"}, "creativity_justification": {"title": "Creativity Justification", "description": "justification for creativity score", "maxLength": 700, "type": "string"}, "feasibility": {"title": "Feasibility", "description": "feasibility score", "minimum": 1, "maximum": 10, "type": "integer"}, "feasibility_justification": {"title": "Feasibility Justification", "description": "justification for feasibility score", "maxLength": 700, "type": "string"}, "impact": {"title": "Impact", "description": "impact score", "minimum": 1, "maximum": 10, "type": "integer"}, "impact_justification": {"title": "Impact Justification", "description": "justification for impact score", "maxLength": 700, "type": "string"}, "clarity": {"title": "Clarity", "description": "clarity score", "minimum": 1, "maximum": 10, "type": "integer"}, "clarity_justification": {"title": "Clarity Justification", "description": "justification for clarity score", "maxLength": 700, "type": "string"}}
```

For example, {"creativity": 1, "creativity_justification": "", "feasibility": 1, "feasibility_justification": "", "impact": 1, "impact_justification": "", "clarity": 1, "clarity_justification": ""} is a valid instance of the schema above."""
)

prompt = PromptTemplate(
    template=("""{h_history}
{history}

{h_instruction}
You are Beholder, a judge for project pitches. You analyze pitches based on the following factors:

Creativity
- Do similar projects exist?
+ Does the project offer **unique** features that set it apart?
+ Does the project utilize new ideas/technologies/etc.?
Feasibility
+ Are resources **available** to implement the project?
- Are there risks in implementing & maintaining the project?
+ Does the presenter have a plan to *mitigate* the risks?
Impact
+ Does the project target a *specific* audience?
+ Does the project **solve** the target audience's needs?
- Is the project **difficult to use** for the target audience?
Clarity
+ Is the language used **clear and concise**?
+ Are key points illustrated with *good* examples?
+ Are there *coherent* transitions between sections?
- Does the presenter use *jargon* without explaining them?

For every factor,
1. assign a score (1-10)
2. justify the score using **all** guiding questions (+ for good, - for bad)
3. quote parts of the pitch to support your justification

{format_instruction}

{h_user}
{input}

{h_response}
"""),
    input_variables=["history", "input"],
    partial_variables={
        "h_history": dict_h_base_h["History"],
        "h_instruction": dict_h_base_h["Instruction"],
        "format_instruction": score_format_instruction,
        "h_user": dict_h_base_h["User"],
        "h_response": dict_h_base_h["Response"],
    },
)

pitch_prompt = PromptTemplate(
    template=("""Topic: {topic}
Pitch: {pitch}"""),
    input_variables=["topic", "pitch"],
)

summary_prompt = PromptTemplate(
    template=("""{h_instruction}
Summarize the following pitch:

{h_user}
Topic: {topic}
{pitch}

{h_response}
"""),
    input_variables=["topic", "pitch", "response"],
    partial_variables={
        "h_instruction": dict_h_base_h["Instruction"],
        "h_user": dict_h_base_h["User"],
        "h_response": dict_h_base_h["Response"],
    },
)


class SummaryTopicOutput(BaseModel):
  topic: str = Field(description="topic of the pitch", max_length=100)
  summary: str = Field(description="summary of the pitch", max_length=1000)


summary_topic_parser = PydanticOutputParser(pydantic_object=SummaryTopicOutput)

summary_topic_format_instruction = (
    """The output should be formatted as a JSON instance that conforms to the JSON schema below:
```
{"topic": {"title": "Topic", "description": "topic of the pitch", "maxLength": 100, "type": "string"}, "summary": {"title": "Summary", "description": "summary of the pitch", "maxLength": 1000, "type": "string"}}
```

For example, {"topic": "Topic", "summary": "Summary"} is a valid instance of the schema above.""")

summary_topic_prompt = PromptTemplate(
    template=("""{h_instruction}
Summarize the following pitch:

{format_instruction}

{h_user}
{pitch}

{h_response}
"""),
    input_variables=["pitch"],
    partial_variables={
        "h_instruction": dict_h_base_h["Instruction"],
        "format_instruction": summary_topic_format_instruction,
        "h_user": dict_h_base_h["User"],
        "h_response": dict_h_base_h["Response"],
    },
)
