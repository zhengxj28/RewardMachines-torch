import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
# from src.networks.rm_network import QRMNet
from src.networks.q_network import DeepQNet
from src.agents.rm_agent import RMAgent
from src.agents.base_rl_agent import BaseRLAgent

from transformers import AutoTokenizer
from transformers import T5EncoderModel

import os
import zhipuai
import replicate
from openai import OpenAI
import openai


class PromptProgAgent(BaseRLAgent):
    """
    to be specified...
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id, use_cuda):
        super().__init__()

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        self.model_params = model_params

        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.device = device

        # TODO(√): defined in RMAgent previously, only one with the help of llm encoder
        # num_policies = self.num_policies
        # num_policies = 1

        
        # TODO: hyper-parameters should be extracted to yaml
        self.task_emb_len = 20 * 768
        self.model_ckpts = "/home/yeshicheng/flan-t5-base"  # use Google Flan-UL2 to encode task
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_ckpts)
        self.llm_encoder = T5EncoderModel.from_pretrained(self.model_ckpts, device_map="auto").to(self.device)
        self.max_new_tokens = 256

        # TODO: use only-one Qnetwork?
        # self.qrm_net = QRMNet(num_features, num_actions, num_policies, model_params).to(device)
        # self.tar_qrm_net = QRMNet(num_features, num_actions, num_policies, model_params).to(device)        
        assert(learning_params.tabular_case == False) # since task embedding only used in neural network
        self.q_net = DeepQNet(num_features+self.task_emb_len, num_actions, model_params).to(device)
        self.tar_q_net = DeepQNet(num_features+self.task_emb_len, num_actions, model_params).to(device)
        self.buffer = ReplayBuffer(num_features+self.task_emb_len, num_actions, learning_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.q_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_params.lr)

    def add_task_embedding(self, state, nl_task):
        
        tokens = self.llm_tokenizer(nl_task, return_tensors="pt").input_ids.to(self.device)
        task_emb = self.llm_encoder(input_ids=tokens).last_hidden_state
        task_emb = task_emb.cpu().view(-1).detach().numpy().flatten()
        
        # TODO: wrong logic, pad task embedding to a constant length
        task_emb = np.pad(  task_emb[:self.task_emb_len],
                            (0, max(0, self.task_emb_len - len(task_emb))), 
                            constant_values=0)

        # print(task_emb.shape, type(task_emb), task_emb)
        # print(state.shape, type(state), state)

        return np.concatenate((task_emb, state))
        # return torch.cat((task_emb.view(-1), state), dim=1)

    def progression(self, nl_task, events):
        # TODO: select better prompts and store them in a file 
        # use ChatGLM API currently
        if events == "":
            return nl_task

        print("nl-task:", nl_task)
        print("events:", " and ".join(events))

        # chat
        # def chat_completions(query):
        #     client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        #     resp = client.chat.completions.create(
        #         model="gpt-3.5-turbo",
        #         messages=[
        #             # {"role": "user", "content": "The task to be solved is: go to a, then go to b, then go to c. Now we have arrived at a and e. what's the updated task?"},
        #             # {"role": "system", "content": "since e is irrelevant to task, so the updated task is: go to b, then go to c"},
        #             # {"role": "user", "content": "The task to be solved is: go to d, then go to e. Now we have arrived at d and k. what's the updated task?"},
        #             # {"role": "system", "content": "since k is irrelevant to task, so the updated task is: go to e"},
        #             # {"role": "user", "content": "The task to be solved is: go to a, then go to b. Now we have arrived at b. what's the updated task?"},
        #             # {"role": "system", "content": "since we should arrive at a before b, so the updated task is: go to a, then go to b"},
        #             # {"role": "user", "content": "The task to be solved is: go to e, then go to f. Now we have arrived at f. what's the updated task?"},
        #             # {"role": "system", "content": "since we should arrive at e before f, so the updated task is: go to e, then go to f"},
        #             # {"role": "user", "content": "The task to be solved is: go to f. Now we have arrived at f. what's the updated task?"},
        #             # {"role": "system", "content": "the updated task is: solved"},
        #             {"role": "user", "content": query}
        #         ]
        #     )
        #     return resp.choices[0].message.content

        # API_SECRET_KEY = "zk-82e65276330c746a100483ab2a732f4d"
        # BASE_URL = "https://flag.smarttrot.com/v1/"
        
        # with open("/home/yeshicheng/RewardMachines-torch/src/agents/prompt.txt", "r") as file:
        #     prompt = file.read()
        # print(prompt)
        # query = "task: {}.\n current_arrived_places: {}.\n output: ".format(nl_task, " and ".join(events))
        
        # answer = chat_completions(prompt+"\n"+query)
        # print(answer)
        # return answer

        with open("/home/yeshicheng/RewardMachines-torch/src/agents/prompt2.txt", "r") as file:
            prompt = file.read()
        query = "task: {}.\n current_arrived_places: {}.\n output: ".format(nl_task, " and ".join(events))

        # zhipuai.api_key = "3d3cbd5204de9557309c45691ebf5634.g5fzRasibOBMwXkk"
        # response = zhipuai.model_api.sse_invoke(
        #     model="chatglm_turbo",
        #     prompt=[
        #         {"role": "user", "content": query}
        #     ],
        #     temperature=0.1,
        #     top_p=0.7,
        #     incremental=True
        # )

        # answer = ""
        # for event in response.events():
        #     answer += event.data
        # print(answer)
        # answer = answer.split(":", 1)[1]
        # answer = answer.strip()
        # return answer

        os.environ["REPLICATE_API_TOKEN"] = "r8_WgrnatJThmzbaFcfb9F2nNB0mlgkrpP1gNuyF"
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "top_p": 0.8,
                "prompt": query,
                "temperature": 0.01,
            },        
        )
        answer = ""
        for word in output:
            answer += word
        return answer

    def learn(self):
        # TODO: nps 被删除后，如何判断是否 done, LLM 判断？algo judge
        # s1, a, s2, rs, nps, _ = self.buffer.sample(self.learning_params.batch_size)
        # done = torch.zeros_like(nps, device=self.device)
        # done[nps == 0] = 1  # NPs[i]==0 means terminal state
        
        s1, a, s2, rs, done = self.buffer.sample(self.learning_params.batch_size)

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.q_net(s1)[ind, :, a.squeeze(1)]
        gamma = self.learning_params.gamma

        # with torch.no_grad():
        #     Q_tar_all = torch.max(self.tar_qrm_net(s2), dim=2)[0]
        #     Q_tar = torch.gather(Q_tar_all, dim=1, index=nps)
        with torch.no_grad():
            Q_tar = torch.max(self.tar_q_net(s2), dim=1)

        loss = torch.Tensor([0.0]).to(self.device)
        # for i in range(self.num_policies):
        #     loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])
        loss += 0.5 * nn.MSELoss()(Q, rs + gamma * Q_tar * (1 - done))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item()}

    def get_action(self, s, eval_mode=False):
        device = self.device
        if eval_mode:
            # TODO: if only one Q-network, eliminate state2policy
            # policy_id = self.state2policy[(self.rm_id_eval, self.u_eval)]
            s = torch.Tensor(s).view(1, -1).to(device)
            # q_value = self.qrm_net(s).detach()[:, policy_id].squeeze()
            q_value = self.q_net(s).detach().squeeze()
            a = torch.argmax(q_value).cpu().item()
        elif random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            # TODO: as previous
            # policy_id = self.state2policy[(self.rm_id, self.u)]
            s = torch.Tensor(s).view(1, -1).to(device)
            # q_value = self.qrm_net(s).detach()[:, policy_id].squeeze()
            q_value = self.q_net(s).detach().squeeze()
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def update(self, s1, a, s2, events, done, eval_mode=False):
        # 这里是用来更新 rm 状态的
        # 把这里改成用来增加样本
        # if not eval_mode:
        #     # Getting rewards and next states for each reward machine to learn
        #     rewards, next_policies = [], []
        #     reward_machines = self.reward_machines
        #     for j in range(len(reward_machines)):
        #         j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
        #         rewards.append(j_rewards)
        #         next_policies.append(j_next_states)
        #     # Mapping rewards and next states to specific policies in the policy bank
        #     rewards = self.map_rewards(rewards)
        #     next_policies = self.map_next_policies(next_policies)
        #     # Adding this experience to the experience replay buffer
        #     self.buffer.add_data(s1, a, s2, rewards, next_policies, done)

        # update current rm state
        # self.update_rm_state(events, eval_mode)
        
        # TODO: judge reward
        reward = 1 if done else -0.01
        self.buffer.add_data(s1, a, s2, reward, done)

    def update_target_network(self):
        self.tar_q_net.load_state_dict(self.q_net.state_dict())

class ReplayBuffer(object):
    # TODO: prioritized replay buffer
    def __init__(self, num_features, num_actions, learning_params, device):
        """
        Create (Prioritized) Replay buffer.
        """
        # self.storage = []

        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        # self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)

        # TODO: remove next policy?
        # self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

        # replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
        #                                                                learning_params.prioritized_replay,
        #                                                                learning_params.prioritized_replay_alpha,
        #                                                                learning_params.prioritized_replay_beta0,
        #                                                                curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

    def add_data(self, s1, a, s2, rewards, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor([rewards])
        
        # self.NPs[idx] = torch.LongTensor(next_policies)

        # actually QRM does not use `done` from the environment
        self.Done[idx] = torch.LongTensor([done])

        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)


    def sample(self, batch_size):
        """Sample a batch of experiences."""
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        # s1 = torch.gather(self.S1, dim=0, index=idxes)
        s1 = self.S1[index]
        a = self.A[index]
        s2 = self.S2[index]
        rs = self.Rs[index]
        # nps = self.NPs[index]
        # return s1, a, s2, rs, nps, None
        done = self.Done[index]
        return s1, a, s2, rs, done
