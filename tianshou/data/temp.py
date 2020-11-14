self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())
result = self.policy(self.data, last_state)#get data.obs
