class TrainableVariableDict(dict):
  def __setitem__(self, k, v):
    if k in self.keys():
      raise ValueError("Key is already present")
    else:
      return super(TrainableVariableDict, self).__setitem__(k, v)

  def update(self, in_dict):
    for (key, value) in in_dict.items():
      self.__setitem__(key, value)
