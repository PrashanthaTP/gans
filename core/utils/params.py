import json 
class Params: 
    def __init__(self):
        self.params = {} 
    # @classmethod
    # def from_dict(cls,d:dict):
    #     params = cls()
    #     for key,value in d.items():
    #         setattr(params,key,value)
        
    #     return params 
    @classmethod
    def from_json(cls,json_file):
        obj = cls()
        with open(json_file,'r') as file: 
            d = json.load(file)
            obj.params = d
            return obj 
    def __getitem__(self,key):
        if key not in self.params:
            raise KeyError(f'Param object has no key {key} ')
        return self.params.get(key)

    def save(self,filepath):
        with open(filepath,'a+') as file:
            json.dump(self.params,file)
        
    