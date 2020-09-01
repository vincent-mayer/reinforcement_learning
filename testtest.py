class ReplayBufferLSTM:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def _init_(self, obs_dim, act_dim, hidden_dim, size, ep_len):
        size = size // ep_len
        self.ep_len = ep_len
        self.act_dim = act_dim
        self.obs_cam_buf = np.zeros(combined_shape(size, combined_shape(ep_len, obs_dim[0])), dtype=np.float32)
        self.obs_robot_buf = np.zeros(combined_shape(size, combined_shape(ep_len, obs_dim[1])), dtype=np.float32)
        self.obs2_cam_buf = np.zeros(combined_shape(size, combined_shape(ep_len, obs_dim[0])), dtype=np.float32)
        self.obs2_robot_buf = np.zeros(combined_shape(size, combined_shape(ep_len, obs_dim[1])), dtype=np.float32)
        self.hidden_in_buf = (np.zeros(combined_shape(size, combined_shape(ep_len, hidden_dim)), dtype=np.float32),
                              np.zeros(combined_shape(size, combined_shape(ep_len, hidden_dim)), dtype=np.float32))
        self.hidden_out_buf = (np.zeros(combined_shape(size, combined_shape(ep_len, hidden_dim)), dtype=np.float32),
                               np.zeros(combined_shape(size, combined_shape(ep_len, hidden_dim)), dtype=np.float32))
        self.seq_len = np.zeros(size, dtype=np.int)
        self.act_buf = np.zeros(combined_shape(size, combined_shape(ep_len, act_dim)), dtype=np.float32)
        self.last_act_buf = np.zeros(combined_shape(size, combined_shape(ep_len+1, act_dim)), dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, ep_len), dtype=np.float32)
        self.int_rew_buf = np.zeros(combined_shape(size, ep_len), dtype=np.float32)
        self.done_buf = np.zeros(combined_shape(size, ep_len), dtype=np.float32)
        self.index, self.sub_index = 0, 0  # sampling index, whenever sampling is ordered
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, hidden_in, hidden_out, obs, act, rew, next_obs, done, int_reward=0.0):
        self.hidden_in_buf[0][self.ptr] = np.array([h[0].squeeze().cpu().detach().numpy() for h in hidden_in])
        self.hidden_in_buf[1][self.ptr] = np.array([h[1].squeeze().cpu().detach().numpy() for h in hidden_in])
        self.hidden_out_buf[0][self.ptr] = np.array([h[0].squeeze().cpu().detach().numpy() for h in hidden_out])
        self.hidden_out_buf[1][self.ptr] = np.array([h[1].squeeze().cpu().detach().numpy() for h in hidden_out])
        self.obs_cam_buf[self.ptr] = np.array([o[0] for o in obs])
        self.obs2_cam_buf[self.ptr] = np.array([o[0] for o in next_obs])
        self.obs_robot_buf[self.ptr] = np.array([o[1] for o in obs])
        self.obs2_robot_buf[self.ptr] = np.array([o[1] for o in next_obs])
        self.act_buf[self.ptr] = act
        self.last_act_buf[self.ptr][1:] = act # memory expensive but saves computation
        self.rew_buf[self.ptr] = rew
        self.int_rew_buf[self.ptr] = int_reward
        self.done_buf[self.ptr] = done
        self.seq_len[self.ptr] = len(act)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def update_int_rew(self, idxs, int_rew, batch_size=32, seq_len=2):
        self.int_rew_buf[self.index: min(self.index + 1, self.size), self.sub_index:min(self.sub_index + seq_len, self.seq_len[self.index])] = int_rew
        self.update_index(batch_size, seq_len, False)
    def update_index(self, batch_size=32, seq_len=2, random=True):
        self.index = min(self.index + batch_size, self.size) % self.size \
            if self.sub_index+seq_len >= self.seq_len[self.index] else self.index
        self.sub_index = 0 if random else min(self.sub_index + seq_len, self.seq_len[self.index]) % self.seq_len[self.index]
        self.index = 0 if random else self.index
    def sample_batch(self, batch_size=32, seq_len=2, random=True, sequential=True):
        idxs = np.random.randint(0, self.size, size=batch_size) if random \
               else np.arange(self.index, min(self.index+batch_size, self.size))
        self.index, self.sub_index = (0, 0) if random else (self.index, self.sub_index)
        sub_idx = [np.random.randint(0, min(self.ep_len-seq_len, self.seq_len[idxs[i]]-seq_len)) for i in range(batch_size)] if random else [self.sub_index]
        sub_idx_range = [range(sub_i, min(sub_i+seq_len, self.seq_len[idxs[i]])) for i,sub_i in enumerate(sub_idx)] \
            if sequential else [np.random.randint(0, self.ep_len, size=seq_len) for i in range(batch_size)]
        batch = dict(hidden_in=(np.array([self.hidden_in_buf[0][idxs[i]][sub_idx[i]] for i in range(batch_size)]),
                                np.array([self.hidden_in_buf[1][idxs[i]][sub_idx[i]] for i in range(batch_size)])),
                     hidden_out=(np.array([self.hidden_out_buf[0][idxs[i]][sub_idx[i]] for i in range(batch_size)]),
                                 np.array([self.hidden_out_buf[1][idxs[i]][sub_idx[i]] for i in range(batch_size)])),
                     obs=(np.array([self.obs_cam_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                          np.array([self.obs_robot_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])),
                     obs2=(np.array([self.obs2_cam_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                          np.array([self.obs2_robot_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])),
                     act=np.array([self.act_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     last_act=np.array([self.last_act_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     rew=np.array([self.rew_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     int_rew=np.array([self.int_rew_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     done=np.array([self.done_buf[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]))
        return {k: (torch.as_tensor(v[0], dtype=torch.float32, device=torch.device('cuda')),
                    torch.as_tensor(v[1], dtype=torch.float32, device=torch.device('cuda')))
                    if k in ['obs', 'obs2']
                    else torch.as_tensor(v, dtype=torch.float32, device=torch.device('cuda'))
                    for k, v in batch.items()}
    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb+') as f:
            pickle.dump(self, f)
        print(f"Buffer dumped to {fn}")