import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
  sys.path.append(module_path)

from dotenv import load_dotenv

load_dotenv(os.path.join(module_path, "server/.env"))

import gradio as gr

from server.config import DEBUG, HOST, PORT, SHARE
from server.demos.receive_video_demo import receive_video_demo

demo = gr.TabbedInterface([
    receive_video_demo,
], ["Receive Video"])

if __name__ == "__main__":
  demo.launch(server_name=HOST, server_port=PORT, debug=DEBUG, share=SHARE)

# # Adapted from https://gist.github.com/tenuki/ff67f87cba5c4c04fd08d9c800437477?permalink_comment_id=4236491#gistcomment-4236491
# async def create_webserver(**kwargs):
#   server_config = uvicorn.Config(**kwargs, host=HOST, log_level=LOGLVL)
#   server = uvicorn.Server(server_config)
#   await server.serve()
# async def main():
#   apps = [create_webserver("main:app")]
#   done, pending = await asyncio.wait(
#       apps,
#       return_when=asyncio.FIRST_COMPLETED,
#   )
#   print("done")
#   print(done)
#   print("pending")
#   print(pending)
#   for pending_task in pending:
#     pending_task.cancel("Another service died, server is shutting down")
# if __name__ == "__main__":
#   try:
#     asyncio.run(main())
#   except Exception as e:
#     print(e)
#     sys.exit(0)
