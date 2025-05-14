from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
import textgrad as tg


# start a server with lm-studio and point it to the right address; here we use the default address. 
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

engine = ChatExternalClient(client=client, model_string='mlabonne/NeuralBeagle14-7B-GGUF')

# Set this engine as the backward engine for TextGrad
tg.set_backward_engine(engine, override=True)

# Create a variable with your prompt
question = tg.Variable("5+3+3+3+(-3)=?", requires_grad=False, role_description="testing connection to KoboldCPP server")

loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                 requires_grad=False,
                                 role_description="system prompt")
                              
loss_fn = tg.TextLoss(loss_system_prompt)

solution = "5+3+3 = 11"
optimizer = tg.TGD([solution])

loss = loss_fn(solution)
print(loss.value)

# this will take a while on CPU. :( 
loss.backward()
optimizer.step()
print(solution.value)



     


