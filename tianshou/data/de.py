import pickle, numpy as np
from tianshou.data import CachedReplayBuffer
buf = CachedReplayBuffer(size=20, cached_buf_n = 2, max_length = 3)

for i in range(3):
    buf.add(obs=[i]*2, act=[i]*2, rew=[i]*2, done=[i]*2, obs_next=[i + 1]*2, info=[{}]*2)
print(buf.obs)
print(buf.done)

# but there are only three valid items, so len(buf) == 3.
print(len(buf))
buf2 = CachedReplayBuffer(size=20, cached_buf_n = 2, max_length = 3)
for i in range(15):
    buf2.add(obs=[i]*2, act=[i]*2, rew=[i]*2, done=[i]*2, obs_next=[i + 1]*2, info=[{}]*2)
print(len(buf2))
print(buf2.obs)

# move buf2's result into buf (meanwhile keep it chronologically)
buf.update(buf2)
print(buf)
# get a random sample from buffer
# the batch_data is equal to buf[incide].
batch_data, indice = buf.sample(batch_size=4)
print(batch_data.obs == buf[indice].obs)
print(len(buf))

