from server.models.llm.tokenizer import response_sep

append = """

RESPONSE:
Topic
* main idea of the pitch
* short (max 100 characters)
=====
"""

prepend = """Summarize the following project pitches.""" + response_sep + f"""INPUT:
Hello, I'm Alex Turner, presenting a project to address the lack of accessible platforms for hands-on learning in space exploration. My solution is the Space Exploration Simulation Platform (SESP). This platform offers an immersive and interactive environment for enthusiasts to conduct experiments, simulate space missions, and enhance their skills. Modeled after successful concepts like Cybersecurity Capture The Flag, SESP provides a user-friendly interface, step-by-step learning, and a live scoreboard for real-time progress tracking. The goal is to empower users to bridge the gap between theoretical knowledge and practical experience, preparing them for the challenges of space exploration. Thank you for considering my proposal.{append}
Space Exploration Simulation Platform""" + response_sep + """INPUT:
"""
